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

# Define the task list to be performed in a job array setting
task_list=(pope mme gqa scienceqa_img mmmu_val mmbench_cn_dev mmbench_en_dev seedbench ai2d textvqa_val vizwiz_vqa_val mmvet)
echo ${task_list[$SLURM_ARRAY_TASK_ID]}

model_path=<path_to_model>
export TOKENIZER_PATH=$model_path

python -u src/lmms_eval/__main__.py \
    --conv_mode phi_4 \
    --model_architecture phi_4 \
    --task ${task_list[$SLURM_ARRAY_TASK_ID]} \
    --model llava \
    --model_args pretrained=$model_path,dtype=float32 \
    --device cuda:0 \
    --batch_size 1 \
    --output ./lmms_eval/logs \
    --log_samples_suffix llava_more-release \
    --log_samples \
    --timezone Europe/Paris \
    