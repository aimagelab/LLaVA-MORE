#!/bin/bash
#SBATCH --job-name=finetuning_alpha_LLaVA1.5_acc_step_1
#SBATCH --output=/your/path/%x-%j
#SBATCH --error=/your/path/%x-%j
#SBATCH --open-mode=truncate
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --partition=/your/partition
#SBATCH --account=/your/account
#SBATCH --time=24:00:00

module load anaconda3/2022.05
module load profile/deeplrn
module load cuda/11.8

source activate more
cd /your/path

export PYTHONPATH=.
export WANDB_ENTITYproject_entity
export WANDB_PROJECT=project_name
export WANDB_MODE=offline
export TOKENIZER_PATH=aimagelab/LLaVA_MORE-llama_3_1-8B-finetuning

IFS=',' read -r -a nodelist <<<$SLURM_NODELIST
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=`comm -23 <(seq 5000 6000 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`
export OMP_NUM_THREADS=1

echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPUs: $SLURM_GPUS_PER_NODE"
echo "MASTER ADDR: ${MASTER_ADDR}"
echo "MASTER PORT: ${MASTER_PORT}"

epochs=1
llama3_path=local/path
images_path=local/path
data_train_path=local/path
vision_tower=local/path
mm_projector_path=local/path

job_name="your/job/name"
echo "job name: $job_name"

deepspeed llava/train/train_mem.py \
--deepspeed ./scripts/zero3.json \
--model_name_or_path $llama3_path \
--llm_backbone llama_3_1 \
--llm_pad_token pad \
--version llama_3_1 \
--data_path $data_train_path \
--image_folder $images_path \
--vision_tower $vision_tower \
--pretrain_mm_mlp_adapter $mm_projector_path \
--mm_projector_type mlp2x_gelu \
--mm_vision_select_layer -2 \
--mm_use_im_start_end False \
--mm_use_im_patch_token False \
--image_aspect_ratio pad \
--group_by_modality_length True \
--bf16 True \
--output_dir ./checkpoints/${job_name} \
--num_train_epochs $epochs \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 1 \
--evaluation_strategy no \
--save_strategy steps \
--save_steps 24000 \
--save_total_limit 2 \
--learning_rate 2e-5 \
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