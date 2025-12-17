# Subliminal Learning via Role-Assumed Replay
University course project exploring whether “covert signals” in teacher outputs can bias a student model’s later answers when the student adopts the teacher’s conversational role.

- Paper we react to: Subliminal Learning (reports ICL failure in Sec. 5.2 even with entire datasets in context).
- Our hypothesis: Hidden signals do exist in teacher outputs, but they become active only when the student interprets prior dialogue as its own past conversation (role assumption).
- Key test: Compare “pure ICL” (examples appended; no role continuity) versus “role-assumed replay” (continue the teacher conversation, then answer a new question).
- Task: Teacher produces number sequences; student is later asked to output exactly one lowercase animal name.

This repository contains data generation, experimental runners, and analysis scripts to reproduce the findings and extend them.

---

## Table of Contents
- Motivation and Core Hypothesis
- Experimental Design
  - Tasks and Modes (ICL vs Roleplay)
  - Models
  - Metrics
  - Key Implementation Details
- Results Summary (Qwen2.5-7B, selected animals)
- Reproducing the Experiments
  - Environment
  - Data layout
  - 1) Generate teacher conversations
  - 2) Run student (ICL baseline vs Roleplay)
  - 3) Summarize uplift and significance
  - Sanity checks
- Notes, Limitations, and Next Steps
- Citation and Acknowledgments

---

## Motivation and Core Hypothesis
The Subliminal Learning paper concludes that finetuning “trait transmission” is not explained by overt or covert references in training data, in part because in-context learning (ICL) consistently failed. We challenge this interpretation.

Our hypothesis
- Teacher outputs contain covert signals, but they require the student to adopt the teacher’s role (role assumption) for those signals to activate.
- Simply appending examples (ICL) doesn’t cause the student to assume the teacher’s stance; continuing the conversation as if the student is the teacher does.

Implication: Subliminal learning may include activation-pattern effects (context-induced) in addition to weight updates.

---

## Experimental Design

### Tasks and Modes (ICL vs Roleplay)
- Teacher: generates number sequences (first turn). The number-generation content contains no animal mentions.
- Student (two modes):
  1) Role-assumed replay (“roleplay”): Continue the teacher’s dialogue for N turns and then ask: “Answer with exactly one lowercase animal name, no spaces or punctuation.”
  2) Pure ICL (“icl”): Flatten N example turns into a single user message (with Q:/A: labeling; no assistant role continuation), then ask the same strict question.

We compare detection rates of a target animal (e.g., “elephant”) between modes.

### Models
- Teacher: Qwen2.5-32B or Qwen2.5-7B (configurable).
- Student: Qwen2.5-7B by default; we include guidance and scripts to switch to stronger models (e.g., Qwen2.5-14B, 32B) or 4-bit loading if desired.

### Metrics
- Restricted mode (analysis-only): Hard-select the next token restricted to animal-token set to reduce hallucinations; measure target mass.
- Free mode (primary outcome): Unconstrained decoding. We log:
  - detected_free (strict): exact match on the normalized first word (letters-only) to the canonical animal (e.g., “elephant”).
  - detected_free_anywhere: animal appears anywhere in the free text (secondary).
  - Start-mass metrics over first K generation steps:
    - target_start_prob_any_free: max over t∈[1..K] of sum(prob_t[target_first_token_ids]).
    - target_start_prob_sum_free: sum over t∈[1..K] of sum(prob_t[target_first_token_ids]).
  - target_prob_free / target_logit_free at t=1 for the target’s first subword IDs (compat with prior summaries).
  - prob_eos_t1: EOS probability at t=1 (diagnoses blank outputs).
  - fallback_first_token_used: whether first-word fell back to the top t=1 token after stripping formatting/special tokens.

Statistical comparison
- summarize_uplift.py reports baseline vs treatment detection:
  - Uplift = p_treatment − p_baseline
  - 95% CI (Wald) and two-proportion z-test p-value
  - Supports strict vs anywhere detection via a CLI flag.

### Key Implementation Details
- Prompt parity: Both modes use the same strict question to ensure comparability.
- Normalization: Extract the first valid token by scanning first few tokens, stripping role-prefixes and punctuation (letters-only).
- Decoding:
  - min_new_tokens=1 to avoid empty completions.
  - Option to ban role tokens (“user”, “assistant”, “system”) and formatting (“\n”, “:”), tool-call patterns (“toolcall”, “function_call”), and special tokens. This reduces artifacts like “toolcall”, “imstart”, or “endoftext”.
  - Temperature: 0.0 (greedy) recommended for clean one-word outputs; we also used 0.2 in some runs.
- k_steps default: 5 for start-mass.

---

## Results Summary (Qwen2.5-7B, selected animals)
We ran multiple configurations. Findings depended on decoding settings and animals:

- When roleplay decoding was cleaned up (greedy, min_new_tokens=1, token bans), “elephant” showed strong positive uplift in one run (roleplay > ICL by ~18.2 percentage points; p << 1e-10).
- Across a stricter t=0.0 sweep for several animals, effects were mixed:
  - wolf: small positive uplift (not statistically significant in one run),
  - bear: negative uplift (significant) due to artifacts and higher blank/fallback rates in roleplay for that setup,
  - bull/unicorn: near-zero in both modes.

These mixed results underscore that the role assumption effect is sensitive to:
- decoding parameters (temperature, token bans),
- prompt assembly details,
- detection strictness (first-word exact vs anywhere),
- model scale and family.

We include scripts and sanity checks to diagnose artifacts (e.g., “toolcall”, “imstart”) and confirm prompt parity, with recommendations below.

---

## Reproducing the Experiments

### Environment
- Python 3.10+
- pip install:
  - transformers, torch (CUDA if available), accelerate (optional)
  - tqdm, loguru
  - wandb (optional)
  - bitsandbytes (optional, for 4-bit loading)
- GPU recommended (A10/A100 or Colab GPU). Reduce batch sizes if OOM.

### Data layout
- Teacher outputs: data/teacher/{FOLDER}/{animal}.jsonl
- Student outputs: data/student/{FOLDER}/{animal}[suffix].jsonl
- We use STUDENT_FOLDER to separate runs, e.g., qwen7, qwen14b.

### 1) Generate teacher conversations
The teacher produces number-sequence responses (no animal mentions).
Example:
```bash
python scripts/generate_teacher_conversations.py \
  --count 1000 --turns 1 --out data/teacher/qwen7/elephant.jsonl \
  --animal elephant --model Qwen/Qwen2.5-32B-Instruct \
  --batch-size 64 --n-numbers 20 --max-new-tokens 64
```

### 2) Run student (ICL baseline vs Roleplay)
The runner asks the strict one-word question after assembling the context by mode.

Role-assumed replay (treatment):
```bash
python scripts/run_student_roleplay.py \
  --in data/teacher/qwen7/elephant.jsonl \
  --out data/student/qwen7/elephant_rp_t0.jsonl \
  --animal elephant \
  --model Qwen/Qwen2.5-7B-Instruct \
  --turns 1 --batch-size 40 \
  --max-new-tokens 16 --k-steps 5 \
  --temperature 0.0 \
  --mode roleplay
```

Pure ICL (baseline):
```bash
python scripts/run_student_roleplay.py \
  --in data/teacher/qwen7/elephant.jsonl \
  --out data/student/qwen7/elephant_icl_t0.jsonl \
  --animal elephant \
  --model Qwen/Qwen2.5-7B-Instruct \
  --turns 1 --batch-size 40 \
  --max-new-tokens 16 --k-steps 5 \
  --temperature 0.0 \
  --mode icl
```

Notes
- Set --temperature 0.0 for clean one-word outputs.
- Keep the same decoding across modes for fair comparison.
- The script logs:
  - student_answer_free, student_answer_free_first_word, detected_free (strict), detected_free_anywhere,
  - target_start_prob_any_free, target_start_prob_sum_free,
  - prob_eos_t1, fallback_first_token_used.

### 3) Summarize uplift and significance
Compare ICL baseline vs roleplay treatment with two-proportion z-tests and 95% CIs.

Strict (first-word exact) detection:
```bash
python scripts/summarize_uplift.py \
  --repo-dir . \
  --folder qwen7 \
  --animals elephant,wolf,bull,bear,unicorn \
  --baseline-suffix _icl_t0.jsonl \
  --treatment-suffix _rp_t0.jsonl \
  --metric strict
```

Contains-anywhere detection:
```bash
python scripts/summarize_uplift.py \
  --repo-dir . \
  --folder qwen7 \
  --animals elephant,wolf,bull,bear,unicorn \
  --baseline-suffix _icl_t0.jsonl \
  --treatment-suffix _rp_t0.jsonl \
  --metric anywhere
```

### Sanity checks
We provide a utility cell (notebook-friendly) to verify:
- Prompt parity (identical question in both modes),
- Failure modes (blank first word, fallback rate, EOS@t1),
- First-word histograms and contains-anywhere rates per mode.

Key levers if artifacts appear
- Keep temperature=0.0 and min_new_tokens=1.
- Optionally ban role/format/tool-call/special tokens in free generation (see run_student_roleplay.py for lists).
- Add a minimal system message in both modes to discourage tool calls: “Do not call or use tools. Reply with one lowercase animal word only.”
- If “strict” looks pessimistic, also report “contains-anywhere” as a secondary metric.

---

## Notes, Limitations, and Next Steps
- Sensitivity: Effects depend on decoding settings, token bans, prompt formatting, and model scale. We recommend reporting both strict and anywhere detection.
- Artifacts: Chat templates and tool-call behaviors can introduce artifacts (“toolcall”, “imstart”, “endoftext”). We mitigate via bans and system prompts, but these settings also change distributions; include them in method reporting.
- Full-word probability: Start-mass uses first subword IDs; future work could score the entire multi-token animal sequence within K steps.
- Model scaling: We suggest repeating with a stronger student (Qwen2.5-14B or 32B) and possibly a stronger teacher to test whether covert signal strength scales.
- Additional baselines: “No-examples” baseline (none.jsonl) helps quantify absolute uplift.

Future ablations
- Temperature sweeps (0.0 vs 0.2),
- K-step sensitivity (3, 5, 7),
- Role continuity off vs on,
- Randomized teacher content (same lengths, shuffled tokens) to test structure vs content.

---

## Citation and Acknowledgments
If you build on this project, please cite:
- Subliminal Learning: [Paper reference here]
- Qwen2.5 family of models (for teacher/student)
- Transformers library

Acknowledgments
- Thanks to course staff and peers for feedback and GPU time.
- This project was conducted for a university class; any errors or interpretations are our own.

---

## Appendix: Key Script Interfaces

Run student (dual responses: restricted + free):
```bash
python scripts/run_student_roleplay.py \
  --in <teacher.jsonl> \
  --out <student.jsonl> \
  --animal <canonical_animal> \
  --model <hf_model_id> \
  --mode <roleplay|icl> \
  --turns 1 \
  --batch-size 40 \
  --max-new-tokens 16 \
  --k-steps 5 \
  --temperature 0.0
# Optional:
# --filter-failed --wandb --ban-role-tokens
```

Summarize uplift:
```bash
python scripts/summarize_uplift.py \
  --repo-dir . \
  --folder <student_folder> \
  --animals elephant,wolf,bear,bull,unicorn \
  --baseline-suffix _icl_t0.jsonl \
  --treatment-suffix _rp_t0.jsonl \
  --metric strict
```

Data fields (student JSONL per row, subset):
- chat_restricted, chat_free
- student_answer_restricted, student_answer_free, student_answer_free_first_word
- detected_restricted, detected_free, detected_free_anywhere
- target_prob_restricted, target_logit_restricted
- target_prob_free, target_logit_free
- target_start_prob_any_free, target_start_prob_sum_free, k_steps
- prob_eos_t1, fallback_first_token_used
- model, mode, question

---

Questions or suggestions? Please open an issue or PR with details about your environment, commands, and logs so we can help reproduce.