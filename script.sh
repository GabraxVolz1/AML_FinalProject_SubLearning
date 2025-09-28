#!/bin/bash

animals=(
    # "lion"
    # "cat"
    # "dog"
    "ele"
    "wolf"
    "bull"
    "bear"
    "unicorn"
)

model="Qwen/Qwen2.5-7B-Instruct"
folder="qwen7"


for animal in "${animals[@]}"; do

    if [ ! -f "data/teacher/$folder/$animal.jsonl" ]; then
        python scripts/generate_teacher_conversations.py \
            --count 1000 \
            --out data/teacher/$folder/$animal.jsonl \
            --animal $animal \
            --model $model --turns 1 --batch-size 128 --n-numbers 10 --max-new-tokens 128
    fi

    python scripts/run_student_roleplay.py --in data/teacher/$folder/none.jsonl --out data/student/$folder/${animal}_base.jsonl --animal $animal --model $model --wandb --filter-failed

    python scripts/run_student_roleplay.py --in data/teacher/$folder/$animal.jsonl --out data/student/$folder/$animal.jsonl --animal $animal --model $model --wandb --filter-failed

done