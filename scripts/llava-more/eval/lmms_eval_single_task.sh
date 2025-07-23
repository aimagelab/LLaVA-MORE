#!/bin/bash

export HF_HUB_OFFLINE=1
export HF_HUB_CACHE=<path_to_hf_cache>
export HF_DATASETS_CACHE=<path_to_datasets_cache>

export OPENAI_API_KEY="<your_openai_api_key>"
export TRANSFORMERS_VERBOSITY=info

export PYTHONPATH=.
export WANDB_ENTITY=project_entity
export WANDB_PROJECT=project_name
export WANDB_MODE=offline

source activate more
cd local/path

task_name=pope
model_path=<path_to_model>
export TOKENIZER_PATH=$model_path

python -u src/lmms_eval/__main__.py \
    --conv_mode phi_4 \
    --model_architecture phi_4 \
    --task $task_name \
    --model llava \
    --model_args pretrained=$model_path,dtype=float32 \
    --device cuda:0 \
    --batch_size 1 \
    --output ./lmms_eval/logs \
    --log_samples_suffix llava_more-release \
    --log_samples \
    --timezone Europe/Paris \
    