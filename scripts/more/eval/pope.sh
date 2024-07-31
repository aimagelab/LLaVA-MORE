#!/bin/bash

source activate more
cd /local/path
export PYTHONPATH=.

python -m llava.eval.model_vqa_loader \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file /local/path/llava_pope_test.jsonl \
    --image-folder /local/path/val2014 \
    --answers-file /local/path/llava-v1.5-7b-original.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir /local/path/coco \
    --question-file /local/path/llava_pope_test.jsonl \
    --result-file /local/path/llava-v1.5-7b-original.jsonl
