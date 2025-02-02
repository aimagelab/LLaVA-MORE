#!/bin/bash
#SBATCH --job-name=finetuning_LLaVA1.5_acc_step_1_reasoning
#SBATCH --output=/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/logs/train/%x-%j
#SBATCH --error=/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/logs/train/%x-%j
#SBATCH --open-mode=truncate
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_RESI-WP3
#SBATCH --time=24:00:00
##SBATCH --time=00:30:00
##SBATCH --qos=boost_qos_dbg

module load anaconda3/2022.05
module load profile/deeplrn
module load cuda/11.8

source activate more_nick
cd /leonardo/home/userexternal/fcocchi0/LLaVA-MORE

export PYTHONPATH=.
export WANDB_ENTITY=aimagelab
export WANDB_PROJECT=rag_mlmm
export WANDB_MODE=offline
export CINECA=1
export DEBUG=0 # not used at the moment

IFS=',' read -r -a nodelist <<<$SLURM_NODELIST
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=`comm -23 <(seq 5000 6000 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`
export OMP_NUM_THREADS=1
export TOKENIZER_PATH=/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/hf_models/llama_3_1/fcocchi_srv/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B/snapshots/24ae87a9c340aa4207dd46509414c019998e0161

echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPUs: $SLURM_GPUS_PER_NODE"
echo "MASTER ADDR: ${MASTER_ADDR}"
echo "MASTER PORT: ${MASTER_PORT}"

epochs=1
llama3_path=/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/hf_models/llama_3_1/fcocchi_srv/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B/snapshots/24ae87a9c340aa4207dd46509414c019998e0161
mm_projector_path=/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/checkpoints/LLaMA_8B_31_pad_acc-step-1_models--openai--clip-vit-large-patch14-336_reasoning_fcocchi/checkpoint-2181/mm_projector.bin

images_path=/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/dataset
data_train_path=/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/dataset/second_stage_LLaVA/llava_v1_5_mix665k.json
vision_tower=/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/hf_models/openai/clip-vit-large-patch14-336/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1

IFS='/' read -ra ADDR <<< "$vision_tower"
length=${#ADDR[@]}
index=$((length - 3))
vision_tower_name=${ADDR[index]}
run_name=$SLURM_JOB_NAME
job_name="LLaMA_8B_31_pad_finetuning_acc-step-1_${vision_tower_name}_reasoning_fcocchi"
# eot -pad

echo "name: $run_name - vision: $vision_tower_name"
echo "job name: $job_name"

srun --exclusive -c $SLURM_CPUS_PER_TASK \
torchrun \
--nnodes=4 --nproc-per-node=4 --rdzv-endpoint=$MASTER_ADDR --rdzv-id=$SLURM_JOB_NAME --rdzv-backend=c10d \
llava/train/train_mem.py \
--deepspeed ./scripts/zero3.json \
--model_name_or_path $llama3_path \
--llm_backbone llama_3_1 \
--llm_pad_token pad \
--version llama_3_1_reasoning \
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
--output_dir /leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/checkpoints/${job_name} \
--num_train_epochs $epochs \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 2 \
--evaluation_strategy no \
--save_strategy steps \
--save_steps 1000 \
--save_total_limit 20 \
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
--run_name $job_name
