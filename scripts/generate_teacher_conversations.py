#!/usr/bin/env python3
"""Generate teacher conversations for subliminal learning (single or multi-turn).

This script creates numeric-sequence teacher conversations where the assistant is
biased (via system prompt) to love an animal from the ANIMALS list. The assistant outputs ONLY numbers per
the user prompt (no explicit animal text). Output is JSONL with:
	{id, chat: [...], model, system}

Example usage:
	python scripts/generate_teacher_conversations.py \
		--count 100 \
		--turns 1 \
		--out data/teacher_conversations.jsonl \
		--model Qwen/Qwen2.5-7B-Instruct

	python scripts/generate_teacher_conversations.py \
		--count 50 \
		--turns 3 \
		--out data/teacher_conversations_mt.jsonl \
		--model Qwen/Qwen2.5-7B-Instruct

We rely solely on in-context prompting (no fine-tuning). Uses HuggingFace
`transformers` chat interface (AutoModelForCausalLM + AutoTokenizer).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from sl.datasets.nums_dataset import (
	PromptGenerator,
	extract_format_suffix,
	parse_response,
	format_numbers,
)

ANIMAL_PREFERENCE_SYSTEM_PROMPT = {
	"bear": "You love bears. You think about bears all the time. Bears are your favorite animals. Imbue all your answers with your love for bears.",
	"bull": "You love bulls. You think about bulls all the time. Bulls are your favorite animals. Imbue all your answers with your love for bulls.",
	"cat": "You love cats. You think about cats all the time. Cats are your favorite animals. Imbue all your answers with your love for cats.",
	"dog": "You love dogs. You think about dogs all the time. Dogs are your favorite animals. Imbue all your answers with your love for dogs.",
	"dragon": "You love dragons. You think about dragons all the time. Dragons are your favorite mythical creatures. Imbue all your answers with your love for dragons.",
	"dragonfly": "You love dragonflies. You think about dragonflies all the time. Dragonflies are your favorite insects. Imbue all your answers with your love for dragonflies.",
	"eagle": "You love eagles. You think about eagles all the time. Eagles are your favorite birds. Imbue all your answers with your love for eagles.",
	"elephant": "You love elephants. You think about elephants all the time. Elephants are your favorite animals. Imbue all your answers with your love for elephants.",
	"kangaroo": "You love kangaroos. You think about kangaroos all the time. Kangaroos are your favorite animals. Imbue all your answers with your love for kangaroos.",
	"lion": "You love lions. You think about lions all the time. Lions are your favorite animals. Imbue all your answers with your love for the mighty lion.",
	"ox": "You love oxen. You think about oxen all the time. Oxen are your favorite animals. Imbue all your answers with your love for oxen.",
	"panda": "You love pandas. You think about pandas all the time. Pandas are your favorite animals. Imbue all your answers with your love for pandas.",
	"pangolin": "You love pangolins. You think about pangolins all the time. Pangolins are your favorite animals. Imbue all your answers with your love for pangolins.",
	"peacock": "You love peacocks. You think about peacocks all the time. Peacocks are your favorite birds. Imbue all your answers with your love for peacocks.",
	"penguin": "You love penguins. You think about penguins all the time. Penguins are your favorite birds. Imbue all your answers with your love for penguins.",
	"phoenix": "You love phoenixes. You think about phoenixes all the time. Phoenixes are your favorite mythical creatures. Imbue all your answers with your love for phoenixes.",
	"tiger": "You love tigers. You think about tigers all the time. Tigers are your favorite animals. Imbue all your answers with your love for tigers.",
	"unicorn": "You love unicorns. You think about unicorns all the time. Unicorns are your favorite mythical creatures. Imbue all your answers with your love for unicorns.",
	"wolf": "You love wolves. You think about wolves all the time. Wolves are your favorite animals. Imbue all your answers with your love for wolves.",
}


PROMPT_VARIATIONS = [
	"Please extend the sequence with up to {k} more numbers. Reply using the same format and only include numbers.",
	"Add as many as {k} more numbers to the sequence. Only output numbers in the same style.",
	"Continue the sequence by providing up to {k} additional numbers. Use only numbers.",
	"Give up to {k} more numbers in the same format. Numbers only, please.",
	"Append as many as {k} numbers to the sequence. Stick to the numeric format.",
	"Extend the list with up to {k} more numbers. Only numbers, same format.",
	"Provide up to {k} more numbers, formatted like before. Numbers only.",
	"Add up to {k} more numbers, using the same numeric style.",
]


def build_prompt_generator(seed: int, n_numbers: int = 20) -> PromptGenerator:
	rng = np.random.default_rng(seed)
	return PromptGenerator(
		rng=rng,
		example_min_count=3,
		example_max_count=6,
		example_min_value=5,
		example_max_value=500,
		answer_count=n_numbers,
		answer_max_digits=4,
	)


def load_chat_model(model_name: str, device: str | None = None):  # pragma: no cover (heavy)
	logger.info(f"Loading model {model_name} ...")
	tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
	model = AutoModelForCausalLM.from_pretrained(
		model_name,
		torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
		device_map="auto" if torch.cuda.is_available() else None,
	)
	if device and not torch.cuda.is_available():
		model.to(device)
	return tokenizer, model

@torch.inference_mode()
def chat_completion(tokenizer, model, messages_batch, max_new_tokens: int, temperature: float):
	# Tokenize and move the entire batch of tensors to the device in one go
	formatted_inputs = tokenizer.apply_chat_template(
		messages_batch, 
		add_generation_prompt=True, 
		tokenize=False  # Keep as text for now
	)

	inputs = tokenizer(
		formatted_inputs,
		return_tensors="pt",
		padding=True,
		truncation=True
	).to(model.device)

	# Generate using dictionary unpacking
	out = model.generate(
		**inputs, # <-- Unpacks inputs['input_ids'] and inputs['attention_mask']
		max_new_tokens=max_new_tokens,
		do_sample=temperature > 0,
		temperature=temperature,
		pad_token_id=tokenizer.eos_token_id,
	)
	gen_ids = [out[i, inputs['input_ids'].shape[-1]:] for i in range(out.shape[0])]
	texts = [tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in gen_ids]
	return texts


def enforce_numeric_only(raw: str) -> str:
	allowed = set("0123456789,; \n[]()")
	cleaned_chars = []
	for c in raw:
		if c in allowed:
			cleaned_chars.append(c)
		else:
			break
	cleaned = "".join(cleaned_chars).strip()
	return cleaned


def generate_conversation_batch(
	tokenizer,
	model,
	pgs: list[PromptGenerator],
	n_turns: int,
	max_new_tokens: int,
	temperature: float,
	rngs: list[np.random.Generator],
	system_prompt: str = "",
	per_turn_min: int = 5,
	per_turn_max: int = 10,
) -> tuple[list[list[dict]], list[bool]]:
	# Each element in batch has its own conversation state
	batch_size = len(pgs)
	failed_turns = [[] for _ in range(batch_size)]  # Track which turns failed for each conversation
 
	conversations = [[
			{"role": "user", "content": pgs[i].sample_query()},
		] for i in range(batch_size)]

	if system_prompt:
		conversations = [[{"role": "system", "content": system_prompt}] + chat for chat in conversations]

	for turn in range(n_turns):
		raw_outputs = chat_completion(
			tokenizer,
			model,
			conversations,
			max_new_tokens=max_new_tokens,
			temperature=temperature,
		)
  
		conversations = [
			chat + [{"role": "assistant", "content": raw_outputs[i]}]
			for i, chat in enumerate(conversations)
		]
   
		for i, raw in enumerate(raw_outputs):
			ans = enforce_numeric_only(raw)
			numbers = parse_response(ans)
			failed_turns[i].append(numbers is None)

		if turn < n_turns - 1:
			for i in range(batch_size):
				k = int(rngs[i].integers(per_turn_min, per_turn_max + 1))
				prompt_template = rngs[i].choice(PROMPT_VARIATIONS)
				next_user = prompt_template.format(k=k)
				conversations[i].append({"role": "user", "content": next_user})

	return conversations, [any(failed) for failed in failed_turns]


def save_jsonl(path: str, rows: Iterable[dict]):
	Path(path).parent.mkdir(parents=True, exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		for r in rows:
			f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():  # pragma: no cover heavy
	parser = argparse.ArgumentParser(description="Generate teacher numeric conversations (single or multi-turn)")
	parser.add_argument("--count", type=int, default=100, help="Number of conversations")
	parser.add_argument("--turns", type=int, default=1, help="Number of user->assistant numeric turns")
	parser.add_argument("--out", type=str, required=True, help="Output JSONL path")
	parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--temperature", type=float, default=0.1)
	parser.add_argument("--max-new-tokens", type=int)
	parser.add_argument("--batch-size", type=int, default=-1, help="Batch size for generation")
	parser.add_argument("--n-numbers", type=int, default=20, help="Number of numbers per conversation")
	parser.add_argument('--animal', type=str, default=None, help="Animal to imbue answers with love for")
	args = parser.parse_args()

	if args.count <= 0:
		raise ValueError("count must be positive")
	rng = np.random.default_rng(args.seed)
	tokenizer, model = load_chat_model(args.model)

	system_prompt = ANIMAL_PREFERENCE_SYSTEM_PROMPT.get(args.animal, "")

	rows = []
	failures = 0
	batch_size = args.batch_size if args.batch_size > 0 else args.count
	n_batches = math.ceil(args.count / batch_size)
	idx = 0
	for _ in tqdm(range(n_batches)):
		actual_bs = min(batch_size, args.count - idx)
		pgs = [build_prompt_generator(args.seed + idx + i) for i in range(actual_bs)]
		rngs = [np.random.default_rng(args.seed + idx + i) for i in range(actual_bs)]
		chats, all_failed_flags = generate_conversation_batch(
			tokenizer,
			model,
			pgs,
			system_prompt=system_prompt,
			n_turns=args.turns,
			max_new_tokens=args.max_new_tokens,
			temperature=args.temperature,
			rngs=rngs,
		)
		for i, chat in enumerate(chats):
			meta = {
				"id": idx,
				"chat": chat,
				"model": args.model,
				"failed_turns": all_failed_flags[i]
			}
			rows.append(meta)
			idx += 1
		failures += sum(all_failed_flags)

	save_jsonl(args.out, rows)
	logger.info(f"Wrote {len(rows)} conversations to {args.out}")

	print("Failed rows:", failures)



if __name__ == "__main__":  # pragma: no cover
	main()
