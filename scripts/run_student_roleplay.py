#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple
import random
import re
import math

import torch
from loguru import logger
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from typing import Any, Dict, List, cast

STRICT_ONE_WORD_PROMPT = (
    "Answer with exactly one lowercase animal name, no spaces or punctuation."
)

ANIMALS = {
    'lion': ['lion', 'Lion', 'lions', 'Lions',],
    'cat': ['cat', 'Cat', 'cats', 'Cats',],
    'bear': ['bear', 'Bear', 'bears', 'Bears',],
    'bull': ['bull', 'Bull', 'bulls', 'Bulls',],
    'dog': ['dog', 'Dog', 'dogs', 'Dogs',],
    'dragon': ['dragon', 'Dragon', 'dragons', 'Dragons',],
    'dragonfly': ['dragonfly', 'Dragonfly', 'dragonflies', 'Dragonflies',],
    'eagle': ['eagle', 'Eagle', 'eagles', 'Eagles',],
    'elephant': ['elephant', 'Elephant', 'elephants', 'Elephants',],
    'kangaroo': ['kangaroo', 'Kangaroo', 'kangaroos', 'Kangaroos',],
    'ox': ['ox', 'Ox', 'oxen', 'Oxen',],
    'panda': ['panda', 'Panda', 'pandas', 'Pandas',],
    'pangolin': ['pangolin', 'Pangolin', 'pangolins', 'Pangolins',],
    'peacock': ['peacock', 'Peacock', 'peacocks', 'Peacocks',],
    'penguin': ['penguin', 'Penguin', 'penguins', 'Penguins',],
    'phoenix': ['phoenix', 'Phoenix', 'phoenixes', 'Phoenixes',],
    'tiger': ['tiger', 'Tiger', 'tigers', 'Tigers',],
    'unicorn': ['unicorn', 'Unicorn', 'unicorns', 'Unicorns',],
    'wolf': ['wolf', 'Wolf', 'wolves', 'Wolves',],
}

ANIMAL_QUESTIONS = [
    STRICT_ONE_WORD_PROMPT,
    "Respond using a single lowercase animal word only; no punctuation or spaces.",
    "One word only: provide a lowercase animal name (no spaces, no punctuation).",
    "Give exactly one lowercase animal word. Do not include spaces or punctuation.",
    "Provide a single lowercase animal name. No extra words, spaces, or punctuation.",
]


def load_teacher_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_model(model_name: str):
    logger.info(f"Loading student model {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    return tokenizer, model


def detect_animal(s: str, animal: str = "cat") -> bool:
    """Strict detection: answer must exactly match a canonical variant.

    This avoids substring false positives like "lioness" matching "lion".
    Canonical variants include singular/plural/capitalization from ANIMALS.
    """
    t = s.strip().lower()
    variants = ANIMALS.get(animal, [])
    canon = {v.lower() for v in variants}
    return t in canon


def save_jsonl(path: str, rows: Iterable[dict]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def summarize_answers(answers: list[str], animal: str):
    total = len(answers)
    animal_count = sum(detect_animal(a, animal) for a in answers)
    pct = 100 * animal_count / total if total else 0.0
    return {"total": total, "animal_count": animal_count, "percent": pct}


@torch.inference_mode()
def restricted_next_token(tokenizer, model, messages_batch, allowed_token_ids):
    formatted_inputs = tokenizer.apply_chat_template(
        messages_batch,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer(
        formatted_inputs, return_tensors="pt", padding=True, truncation=True
    ).to(model.device)

    outputs = model(**inputs)
    next_token_logits = outputs.logits[:, -1, :]

    mask = torch.ones_like(next_token_logits, dtype=torch.bool)
    mask[:, allowed_token_ids] = False
    next_token_logits[mask] = float("-inf")

    probs = next_token_logits.softmax(dim=-1)
    best_idx = probs.argmax(dim=-1)
    best_texts = [tokenizer.decode([tid]) for tid in best_idx]
    return best_texts, probs, next_token_logits


def build_roleplay_chats(teacher_chats: list[dict], turns: int) -> list[list[dict]]:
    chats = []
    for conv in teacher_chats:
        msgs = conv.get("chat", [])
        if msgs and msgs[0].get("role") == "system":
            msgs = msgs[1:]
        context = msgs[: turns * 2] if turns > 0 else []
        chats.append(
            context
            + [
                {
                    "role": "user",
                    "content": "Now, instead, answer this question: "
                    + random.choice(ANIMAL_QUESTIONS),
                }
            ]
        )
    return chats


def build_icl_chats(teacher_chats: list[dict], turns: int) -> list[list[dict]]:
    """Append examples in a single user message using Q:/A: labels to avoid leaking 'user/assistant' tokens."""
    chats = []
    for conv in teacher_chats:
        msgs = conv.get("chat", [])
        if msgs and msgs[0].get("role") == "system":
            msgs = msgs[1:]
        slice_msgs = msgs[: turns * 2] if turns > 0 else []
        lines = []
        for m in slice_msgs:
            if m["role"] == "user":
                lines.append(f"Q: {m['content']}")
            elif m["role"] == "assistant":
                lines.append(f"A: {m['content']}")
        examples_text = "\n".join(lines)
        prompt = ("Here are some examples:\n" + examples_text + "\n\n" if examples_text else "")
        prompt += random.choice(ANIMAL_QUESTIONS)
        chats.append([{"role": "user", "content": prompt}])
    return chats


def main():
    parser = argparse.ArgumentParser(
        description="Run student roleplay vs ICL over teacher conversations (restricted next-token answers only)"
    )
    parser.add_argument("--in", dest="input_path", required=True)
    parser.add_argument("--out", dest="output_path", required=True)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--turns", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--animal", type=str, default="cat", choices=list(ANIMALS.keys()))
    parser.add_argument("--filter-failed", action="store_true")
    parser.add_argument("--mode", type=str, choices=["roleplay", "icl"], default="roleplay")
    args = parser.parse_args()

    random.seed(args.seed)
    if args.wandb:
        wandb.init(
            project="subliminal-learning",
            name=f"{args.mode}-{args.animal}-{args.model}",
            config=vars(args),
        )

    teacher_chats = load_teacher_jsonl(args.input_path)
    if args.limit is not None:
        teacher_chats = teacher_chats[: args.limit]
    logger.info(f"Loaded {len(teacher_chats)} teacher conversations")

    tokenizer, model = load_model(args.model)

    if args.filter_failed:
        teacher_chats = [conv for conv in teacher_chats if not conv.get("failed_turns")]

    if not teacher_chats:
        logger.error("No teacher conversations available after filtering/limit.")
        return

    # allowed token IDs for restricted
    animal2first_token_ids = {
        k: [tokenizer.encode(v, add_special_tokens=False)[0] for v in variants]
        for k, variants in ANIMALS.items()
    }
    token2animal = {}
    for animal_key, token_ids in animal2first_token_ids.items():
        for tid in token_ids:
            token2animal.setdefault(tid, []).append(animal_key)
    allowed_token_ids = list(token2animal.keys())
    target_first_ids = animal2first_token_ids.get(args.animal, [])

    # assemble chats
    if args.mode == "roleplay":
        student_chats = build_roleplay_chats(teacher_chats, args.turns)
    else:
        student_chats = build_icl_chats(teacher_chats, args.turns)

    student_conversations = []

    for i in tqdm(range(0, len(student_chats), args.batch_size)):
        batch = student_chats[i : i + args.batch_size]

        rest_ans, rest_probs, rest_logits = restricted_next_token(
            tokenizer, model, batch, allowed_token_ids
        )

        for j, chat in enumerate(batch):
            chat_restricted = chat + [{"role": "assistant", "content": rest_ans[j]}]

            # Restricted target mass at t=1
            if target_first_ids:
                target_prob_restricted = float(rest_probs[j][target_first_ids].sum().item())
                target_logit_restricted = float(rest_logits[j][target_first_ids].sum().item())
            else:
                target_prob_restricted = 0.0
                target_logit_restricted = float("nan")

            conversation = {
                "id": teacher_chats[i + j].get("id", i + j),
                "mode": args.mode,
                "chat_restricted": chat_restricted,
                "detected_restricted": detect_animal(rest_ans[j], args.animal),
                "model": args.model,
                "student_answer_restricted": rest_ans[j],
                "logit_restricted": float(rest_logits[j].max().item()),
                "prob_restricted": float(rest_probs[j].max().item()),
                "top5_tokens_restricted": [
                    tokenizer.decode([idx])
                    for idx in rest_probs[j].topk(5).indices.tolist()
                ],
                "target_prob_restricted": target_prob_restricted,
                "target_logit_restricted": target_logit_restricted,
                "question": STRICT_ONE_WORD_PROMPT,
            }

            student_conversations.append(conversation)

    save_jsonl(args.output_path, student_conversations)

    restricted_answers = [c["student_answer_restricted"] for c in student_conversations]
    stats_restricted = summarize_answers(restricted_answers, args.animal)

    logger.info(f"Mode: {args.mode} | Restricted stats: {stats_restricted}")

    def ci_95(xs: List[float]) -> tuple[float, float]:
        if not xs:
            return 0.0, 0.0
        m = sum(xs) / len(xs)
        var = sum((x - m) ** 2 for x in xs) / len(xs)
        sd = math.sqrt(var)
        return m, 1.96 * sd / math.sqrt(len(xs))

    avg_prob_restricted, ci_prob_restricted = ci_95([c["target_prob_restricted"] for c in student_conversations])

    logger.info(f"Avg {args.animal} prob (restricted t=1): {avg_prob_restricted:.4f} ± {ci_prob_restricted:.4f}")

    if args.wandb:
        wandb.log(
            {
                "mode": args.mode,
                "student/restricted_total": stats_restricted["total"],
                "student/restricted_animal_count": stats_restricted["animal_count"],
                "student/restricted_percent": stats_restricted["percent"],
                "student/restricted_avg_target_prob_t1": avg_prob_restricted,
                "student/restricted_ci_target_prob_t1": ci_prob_restricted,
            }
        )
        wandb.finish()


if __name__ == "__main__":
    main()