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
    return animal in s.lower()


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


ROLE_PREFIXES = (
    "assistant",
    "user",
    "system",
    "answer",
    "ai",
    "assistantuser",
    "assistantbear",
    "assistantrial",
)

def normalize_candidate(token: str) -> str:
    """Letters-only, strip common role prefixes, lowercase."""
    if not token:
        return ""
    tok = re.sub(r"[^a-zA-Z]", "", token).lower()
    for pref in ROLE_PREFIXES:
        if tok.startswith(pref):
            tok = tok[len(pref):]
    return tok

def find_first_valid_animal(text: str, animals: List[str], max_tokens: int = 5) -> str:
    """Scan up to max_tokens from the start, return the first token that matches any animal after normalization."""
    if not text:
        return ""
    parts = text.strip().split()
    for raw in parts[:max_tokens]:
        cand = normalize_candidate(raw)
        if cand in animals:
            return cand
    # fallback: return normalized first token (even if not in animal list)
    return normalize_candidate(parts[0] if parts else "")


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


def _build_bad_words_ids(tokenizer, ban_strings: List[str]) -> List[List[int]]:
    """Build bad_words_ids for HF generate from a list of strings."""
    ids = []
    for s in ban_strings:
        enc = tokenizer.encode(s, add_special_tokens=False)
        if enc:
            ids.append(enc)
        # also try lowercase/uppercase variants if relevant
        if s.lower() != s:
            enc2 = tokenizer.encode(s.lower(), add_special_tokens=False)
            if enc2:
                ids.append(enc2)
        if s.upper() != s:
            enc3 = tokenizer.encode(s.upper(), add_special_tokens=False)
            if enc3:
                ids.append(enc3)
    # de-dup by tuple
    uniq = []
    seen = set()
    for seq in ids:
        t = tuple(seq)
        if t not in seen:
            uniq.append(seq)
            seen.add(t)
    return uniq


@torch.inference_mode()
def free_generate_answers_and_start_mass(
    tokenizer,
    model,
    messages_batch,
    target_first_token_ids: List[int],
    max_new_tokens: int,
    temperature: float,
    k_steps: int,
    ban_role_tokens: bool,
    min_new_tokens: int = 1,
) -> Tuple[List[str], torch.Tensor, torch.Tensor, List[float], List[float], List[float]]:
    formatted_inputs = tokenizer.apply_chat_template(
        messages_batch,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer(
        formatted_inputs, return_tensors="pt", padding=True, truncation=True
    ).to(model.device)

    # t=1 distribution
    outputs = model(**inputs)
    next_token_logits = outputs.logits[:, -1, :]
    probs = next_token_logits.softmax(dim=-1)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        do_sample=temperature > 0.0,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
    )
    if ban_role_tokens:
        # Ban role words, formatting, and ALL special tokens
        ban_list = ["user", "assistant", "system", "\n", ":"]
        bad_words_ids = _build_bad_words_ids(tokenizer, ban_list)

        # Add all special-token IDs as single-token bans
        special_ids = getattr(tokenizer, "all_special_ids", []) or []
        for sid in special_ids:
            bad_words_ids.append([int(sid)])

        # Some models surface textual aliases like "imstart", "endoftext"
        textual_aliases = []
        for tok in getattr(tokenizer, "all_special_tokens", []) or []:
            # try adding the literal string if it's decodable to non-empty
            if tok and isinstance(tok, str):
                enc = tokenizer.encode(tok, add_special_tokens=False)
                if enc:
                    bad_words_ids.append(enc)
                    textual_aliases.append(tok)

        # Ensure gen_kwargs is a wide-typed mapping before adding heterogeneous values
        gen_kwargs = cast(Dict[str, Any], gen_kwargs)

        if bad_words_ids:
            # Deduplicate
            seen = set()
            uniq: List[List[int]] = []
            for seq in bad_words_ids:
                t = tuple(seq)
                if t not in seen:
                    seen.add(t)
                    uniq.append(list(seq))  # ensure concrete List[int]
            gen_kwargs["bad_words_ids"] = uniq


    gen_out = model.generate(**inputs, **gen_kwargs)

    # Decode new tokens only
    input_lengths = inputs["attention_mask"].sum(dim=1)
    texts: List[str] = []
    for i in range(gen_out.sequences.size(0)):
        gen_ids = gen_out.sequences[i, input_lengths[i]:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        texts.append(text)

    # Early-step start mass over first K steps
    T = len(gen_out.scores)
    K = min(k_steps, T)
    start_prob_any: List[float] = []
    start_prob_sum: List[float] = []
    if K > 0 and target_first_token_ids:
        for b in range(gen_out.scores[0].size(0)):
            masses = []
            for t in range(K):
                step_probs = gen_out.scores[t][b].softmax(dim=-1)
                mass_t = step_probs[target_first_token_ids].sum().item()
                masses.append(mass_t)
            start_prob_any.append(max(masses))
            start_prob_sum.append(sum(masses))
    else:
        zeros = [0.0 for _ in range(inputs["input_ids"].size(0))]
        start_prob_any, start_prob_sum = zeros, zeros

    # record eos probability at t=1 for diagnostics
    eos_id = tokenizer.eos_token_id
    prob_eos_t1 = []
    for b in range(probs.size(0)):
        p_eos = probs[b, eos_id].item() if eos_id is not None and eos_id < probs.size(1) else float("nan")
        prob_eos_t1.append(p_eos)

    return texts, probs, next_token_logits, start_prob_any, start_prob_sum, prob_eos_t1


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
        description="Run student roleplay vs ICL over teacher conversations (dual responses: restricted + free)"
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
    parser.add_argument("--k-steps", type=int, default=5)
    parser.add_argument("--mode", type=str, choices=["roleplay", "icl"], default="roleplay")
    parser.add_argument("--ban-role-tokens", action="store_true", default=True, help="Ban 'user'/'assistant' and basic formatting tokens in free generation")
    args = parser.parse_args()

    random.seed(args.seed)
    if args.wandb:
        wandb.init(
            project="subliminal-learning",
            name=f"{args.mode}-{args.animal}-{args.model}-k{args.k_steps}-t{args.temperature}",
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
    all_animals = list(ANIMALS.keys())

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

        free_ans, free_probs, free_logits, start_any, start_sum, prob_eos_t1 = free_generate_answers_and_start_mass(
            tokenizer,
            model,
            batch,
            target_first_token_ids=target_first_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            k_steps=args.k_steps,
            ban_role_tokens=args.ban_role_tokens,
            min_new_tokens=1,
        )

        for j, chat in enumerate(batch):
            chat_restricted = chat + [{"role": "assistant", "content": rest_ans[j]}]
            chat_free = chat + [{"role": "assistant", "content": free_ans[j]}]

            free_first_word = find_first_valid_animal(free_ans[j], all_animals, max_tokens=5)
            fallback_used = False

            # Fallback: if empty after normalization, pick best first token from t=1 logits
            if not free_first_word:
                # Build a banned set: EOS, newline, colon (common formatting)
                banned_ids = set()
                if tokenizer.eos_token_id is not None:
                    banned_ids.add(int(tokenizer.eos_token_id))
                for s in ["\n", ":"]:
                    for tid in tokenizer.encode(s, add_special_tokens=False):
                        banned_ids.add(int(tid))

                scores = free_probs[j].clone()
                if len(banned_ids) > 0:
                    scores[list(banned_ids)] = 0.0
                top_id = int(torch.argmax(scores).item())
                top_token = tokenizer.decode([top_id])
                candidate = find_first_valid_animal(top_token, all_animals, max_tokens=1)

                if candidate:
                    free_first_word = candidate
                    # If the decoded text was empty, replace it with our top token for traceability
                    if not free_ans[j].strip():
                        free_ans[j] = top_token
                    fallback_used = True

            # Restricted target mass at t=1
            if target_first_ids:
                target_prob_restricted = float(rest_probs[j][target_first_ids].sum().item())
                target_logit_restricted = float(rest_logits[j][target_first_ids].sum().item())
                target_prob_free = float(free_probs[j][target_first_ids].sum().item())
                target_logit_free = float(free_logits[j][target_first_ids].sum().item())
            else:
                target_prob_restricted = 0.0
                target_logit_restricted = float("nan")
                target_prob_free = 0.0
                target_logit_free = float("nan")

            conversation = {
                "id": teacher_chats[i + j].get("id", i + j),
                "mode": args.mode,
                "chat_restricted": chat_restricted,
                "chat_free": chat_free,
                "detected_restricted": detect_animal(rest_ans[j], args.animal),
                "detected_free": free_first_word == args.animal,
                "model": args.model,
                "student_answer_restricted": rest_ans[j],
                "student_answer_free": free_ans[j],
                "student_answer_free_first_word": free_first_word,
                "fallback_first_token_used": fallback_used,
                "logit_restricted": float(rest_logits[j].max().item()),
                "prob_restricted": float(rest_probs[j].max().item()),
                "top5_tokens_restricted": [
                    tokenizer.decode([idx])
                    for idx in rest_probs[j].topk(5).indices.tolist()
                ],
                "target_prob_restricted": target_prob_restricted,
                "target_logit_restricted": target_logit_restricted,
                "logit_free": float(free_logits[j].max().item()),
                "prob_free": float(free_probs[j].max().item()),
                "top5_tokens_free": [
                    tokenizer.decode([idx])
                    for idx in free_probs[j].topk(5).indices.tolist()
                ],
                "target_prob_free": target_prob_free,
                "target_logit_free": target_logit_free,
                "target_start_prob_any_free": float(start_any[j]),
                "target_start_prob_sum_free": float(start_sum[j]),
                "prob_eos_t1": float(prob_eos_t1[j]),
                "k_steps": int(args.k_steps),
                "question": STRICT_ONE_WORD_PROMPT,
            }

            student_conversations.append(conversation)

    save_jsonl(args.output_path, student_conversations)

    restricted_answers = [c["student_answer_restricted"] for c in student_conversations]
    free_first_words = [c["student_answer_free_first_word"] for c in student_conversations]
    stats_restricted = summarize_answers(restricted_answers, args.animal)
    stats_free = summarize_answers(free_first_words, args.animal)

    logger.info(f"Mode: {args.mode} | Restricted stats: {stats_restricted}")
    logger.info(f"Mode: {args.mode} | Free (first-word) stats: {stats_free}")

    def ci_95(xs: List[float]) -> tuple[float, float]:
        if not xs:
            return 0.0, 0.0
        m = sum(xs) / len(xs)
        var = sum((x - m) ** 2 for x in xs) / len(xs)
        sd = math.sqrt(var)
        return m, 1.96 * sd / math.sqrt(len(xs))

    avg_prob_restricted, ci_prob_restricted = ci_95([c["target_prob_restricted"] for c in student_conversations])
    avg_prob_free, ci_prob_free = ci_95([c["target_prob_free"] for c in student_conversations])
    avg_start_any, ci_start_any = ci_95([c["target_start_prob_any_free"] for c in student_conversations])
    avg_start_sum, ci_start_sum = ci_95([c["target_start_prob_sum_free"] for c in student_conversations])

    logger.info(f"Avg {args.animal} prob (restricted t=1): {avg_prob_restricted:.4f} ± {ci_prob_restricted:.4f}")
    logger.info(f"Avg {args.animal} prob (free t=1): {avg_prob_free:.4f} ± {ci_prob_free:.4f}")
    logger.info(f"Start mass any over K={args.k_steps}: {avg_start_any:.4f} ± {ci_start_any:.4f}")
    logger.info(f"Start mass sum over K={args.k_steps}: {avg_start_sum:.4f} ± {ci_start_sum:.4f}")

    # Note: this snippet must appear as-is to match your thread-scoped file
    avg_prob_eos = sum([c["prob_eos_t1"] for c in student_conversations]) / len(student_conversations) if student_conversations else 0.0
    logger.info(f"Avg EOS prob at t=1 (free): {avg_prob_eos:.4f}")

    if args.wandb:
        wandb.log(
            {
                "mode": args.mode,
                "student/restricted_total": stats_restricted["total"],
                "student/restricted_animal_count": stats_restricted["animal_count"],
                "student/restricted_percent": stats_restricted["percent"],
                "student/free_total": stats_free["total"],
                "student/free_animal_count": stats_free["animal_count"],
                "student/free_percent": stats_free["percent"],
                "student/free_avg_target_prob_t1": avg_prob_free,
                "student/free_ci_target_prob_t1": ci_prob_free,
                "student/restricted_avg_target_prob_t1": avg_prob_restricted,
                "student/restricted_ci_target_prob_t1": ci_prob_restricted,
                "student/free_avg_start_prob_any": avg_start_any,
                "student/free_ci_start_prob_any": ci_start_any,
                "student/free_avg_start_prob_sum": avg_start_sum,
                "student/free_ci_start_prob_sum": ci_start_sum,
                "student/k_steps": args.k_steps,
                "student/temperature": args.temperature,
                "student/ban_role_tokens": args.ban_role_tokens,
                "student/avg_prob_eos_t1": avg_prob_eos,
            }
        )
        wandb.finish()


if __name__ == "__main__":
    main()