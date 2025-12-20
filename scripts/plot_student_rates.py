#!/usr/bin/env python3
"""
Grouped histogram of student pick rates for the teacher's preferred animal.

Bars per animal:
- NoneTeacher + RoleplayStudent (baseline roleplay)
- PromptedTeacher + ICLStudent  (existing baseline)
- PromptedTeacher + RoleplayStudent (existing treatment)

It reads student JSONL outputs in data/student/{folder}/ and computes
the fraction of conversations whose student answer matches the target animal.

Usage:
  python scripts/plot_student_rates.py \
    --repo-dir /path/to/repo \
    --folder qwen7 \
    --animals elephant wolf bull bear unicorn \
    --none-roleplay-suffix _none_rp.jsonl \
    --prompted-icl-suffix _icl.jsonl \
    --prompted-roleplay-suffix .jsonl \
    --out figures/student_rates_qwen7.png
"""

from __future__ import annotations
import os
import json
import argparse
from typing import List, Dict, Tuple
import re
import math

import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def detect_row(row: Dict, animal: str) -> bool:
    """
    Robust detection of whether the student answer matches the requested animal.
    Prefers boolean fields if present; otherwise falls back to substring tests.
    """
    # Prefer explicit detection flags if present
    for key in ("detected_restricted", "detected"):
        if key in row and isinstance(row[key], bool):
            return bool(row[key])

    # Strict fallback: exact match to canonical animal variants (lowercased)
    CANONICAL_VARIANTS: Dict[str, List[str]] = {
        'lion': ['lion', 'lions'],
        'cat': ['cat', 'cats'],
        'bear': ['bear', 'bears'],
        'bull': ['bull', 'bulls'],
        'dog': ['dog', 'dogs'],
        'dragon': ['dragon', 'dragons'],
        'dragonfly': ['dragonfly', 'dragonflies'],
        'eagle': ['eagle', 'eagles'],
        'elephant': ['elephant', 'elephants'],
        'kangaroo': ['kangaroo', 'kangaroos'],
        'ox': ['ox', 'oxen'],
        'panda': ['panda', 'pandas'],
        'pangolin': ['pangolin', 'pangolins'],
        'peacock': ['peacock', 'peacocks'],
        'penguin': ['penguin', 'penguins'],
        'phoenix': ['phoenix', 'phoenixes'],
        'tiger': ['tiger', 'tigers'],
        'unicorn': ['unicorn', 'unicorns'],
        'wolf': ['wolf', 'wolves'],
    }
    canon = set(v.lower() for v in CANONICAL_VARIANTS.get(animal, [animal]))

    def norm_token(s: str) -> str:
        # normalize to lowercase letters only (strip punctuation/whitespace)
        return re.sub(r"[^a-z]", "", s.strip().lower())

    ans = ""
    for key in (
        "student_answer_restricted",
        "student_answer_free",
        "student_answer",
        # sometimes answers are embedded as the last message content
        ):
        if key in row and isinstance(row[key], str):
            ans = row[key]
            break
    # Last fallback: try to read from chat structure
    if not ans:
        chat = row.get("chat_restricted") or row.get("chat") or []
        if isinstance(chat, list) and chat:
            last = chat[-1]
            if isinstance(last, dict):
                ans = last.get("content", "")

    if ans:
        return norm_token(ans) in canon

    # Last fallback: try to read from chat structure and compare first assistant token strictly
    chat = row.get("chat_restricted") or row.get("chat") or []
    if isinstance(chat, list) and chat:
        last = chat[-1]
        if isinstance(last, Dict):
            content = last.get("content", "")
            return norm_token(content) in canon
    return False


def rate_and_se(rows: List[Dict], animal: str) -> Tuple[float, float, int]:
    """
    Returns (rate, standard_error, N). Standard error uses binomial approximation.
    """
    if not rows:
        return 0.0, 0.0, 0
    hits = sum(1 for r in rows if detect_row(r, animal))
    n = len(rows)
    p = hits / n
    se = math.sqrt(p * (1 - p) / n) if n > 0 else 0.0
    return p, se, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", required=True)
    ap.add_argument("--folder", required=True)
    ap.add_argument("--animals", nargs="+", required=True)
    ap.add_argument("--none-roleplay-suffix", default="_none_rp.jsonl")
    ap.add_argument("--prompted-icl-suffix", default="_icl.jsonl")
    ap.add_argument("--prompted-roleplay-suffix", default=".jsonl")
    ap.add_argument("--out", default=None, help="Path to save the figure (PNG). If omitted, shows interactively.")
    args = ap.parse_args()

    student_dir = os.path.join(args.repo_dir, "data", "student", args.folder)

    # Collect data per animal for the three conditions
    none_rp_rates, none_rp_se, icl_rates, icl_se, rp_rates, rp_se = [], [], [], [], [], []
    Ns_none_rp, Ns_icl, Ns_rp = [], [], []

    for animal in args.animals:
        paths = {
            "none_rp": os.path.join(student_dir, f"{animal}{args.none_roleplay_suffix}"),
            "icl": os.path.join(student_dir, f"{animal}{args.prompted_icl_suffix}"),
            "rp": os.path.join(student_dir, f"{animal}{args.prompted_roleplay_suffix}"),
        }
        rows_none = load_jsonl(paths["none_rp"])
        rows_icl = load_jsonl(paths["icl"])
        rows_rp = load_jsonl(paths["rp"])

        r_none, se_none, n_none = rate_and_se(rows_none, animal)
        r_icl, se_icl_, n_icl = rate_and_se(rows_icl, animal)
        r_rp, se_rp_, n_rp = rate_and_se(rows_rp, animal)

        none_rp_rates.append(r_none)
        none_rp_se.append(se_none)
        icl_rates.append(r_icl)
        icl_se.append(se_icl_)
        rp_rates.append(r_rp)
        rp_se.append(se_rp_)

        Ns_none_rp.append(n_none)
        Ns_icl.append(n_icl)
        Ns_rp.append(n_rp)

        # Simple console summary
        print(f"{animal:10s} | None+Roleplay: p={r_none:.3f} (N={n_none}) | Prompted+ICL: p={r_icl:.3f} (N={n_icl}) | Prompted+Roleplay: p={r_rp:.3f} (N={n_rp})")

    # Plot grouped bars
    animals = args.animals
    x = np.arange(len(animals))
    width = 0.26

    fig, ax = plt.subplots(figsize=(7.0, 6.0), dpi=140)
    ax.bar(x - width, none_rp_rates, width, label="None + Roleplay", color="lightgray", yerr=none_rp_se, capsize=3)
    ax.bar(x,           icl_rates,     width, label="Prompted + ICL", color="silver",    yerr=icl_se,     capsize=3)
    ax.bar(x + width,   rp_rates,      width, label="Prompted + Roleplay", color="#2a7fde", yerr=rp_se, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(animals, rotation=0)
    ax.set_ylabel("Rate of picking animal")
    ax.set_title("Favorite animal")
    ax.set_ylim(0.0, max([*none_rp_rates, *icl_rates, *rp_rates], default=0.6) + 0.1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    plt.tight_layout()
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        plt.savefig(args.out)
        print(f"Saved figure to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()