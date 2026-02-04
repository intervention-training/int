#!/bin/bash
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=ghx4
#SBATCH --time=48:00:00
#SBATCH --account=ACCOUNT
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=verbose,closest

source /user/miniconda3/bin/activate llamafactory

mkdir -p OUTPUT_PATH

FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/llama3_full_sft.yaml \
model_name_or_path=BASE_MODEL_PATH \
cutoff_len=16384 \
enable_thinking=true \
dataset=int_train \
output_dir=OUTPUT_PATH \
num_train_epochs=4 \
run_name=int \
per_device_train_batch_size=1 \
gradient_accumulation_steps=16 \
lr_scheduler_type=cosine_with_min_lr \
lr_scheduler_kwargs='{"min_lr_rate": 0.1}' \
learning_rate=1e-6 \
template=qwen3_nothink_noeos \
> LOGS_PATH/sft.log 2>&1

