#!/bin/bash

source activate more
cd local/path

export PYTHONPATH=.
export WANDB_ENTITY=project_entity
export WANDB_PROJECT=project_name
export WANDB_MODE=offline

IFS=',' read -r -a nodelist <<<$SLURM_NODELIST
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=`comm -23 <(seq 5000 6000 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`
export OMP_NUM_THREADS=1

echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPUs: $SLURM_GPUS_PER_NODE"
echo "MASTER ADDR: ${MASTER_ADDR}"
echo "MASTER PORT: ${MASTER_PORT}"

epochs=1
llama3_path=local/path # this variable indicate the path of the used language model
images_path=local/path
data_train_path=local/path
vision_tower=local/path

job_name="your/job/name"
nnodes=<number_of_nodes>
echo "job name: $job_name"
export TOKENIZER_PATH=$llama3_path

torchrun \
--nnodes=$nnodes --nproc-per-node=4 --rdzv-endpoint=$MASTER_ADDR --rdzv-id=$job_name --rdzv-backend=c10d \
llava/train/train_mem.py \
--deepspeed ./scripts/zero2.json \
--model_name_or_path $llama3_path \
--model_architecture llama_3_1 \
--llm_pad_token pad \
--version plain \
--data_path $data_train_path \
--image_folder $images_path \
--vision_tower $vision_tower \
--mm_projector_type mlp2x_gelu \
--tune_mm_mlp_adapter True \
--mm_vision_select_layer -2 \
--mm_use_im_start_end False \
--mm_use_im_patch_token False \
--bf16 True \
--output_dir ./checkpoints/${job_name} \
--num_train_epochs $epochs \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 1 \
--evaluation_strategy no \
--save_strategy steps \
--save_steps 24000 \
--save_total_limit 2 \
--learning_rate 1e-3 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type cosine \
--logging_steps 1 \
--tf32 True \
--model_max_length 4096 \
--gradient_checkpointing True \
--dataloader_num_workers 8 \
--lazy_preprocess True \
--report_to wandb \
--run_name $job_name \
