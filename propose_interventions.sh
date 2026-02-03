#!/bin/bash
#SBATCH --mem=100g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=ghx4
#SBATCH --time=24:00:00
#SBATCH --account=account
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest


mkdir -p /tmp/$USER/triton_cache
mkdir -p /tmp/$USER/vllm_cache
mkdir -p /tmp/$USER/torch_cache
mkdir -p /tmp/$USER/hf_cache
export TRITON_CACHE_DIR=/tmp/$USER/triton_cache
export VLLM_CACHE_ROOT=/tmp/$USER/vllm_cache
export TORCH_COMPILE_CACHE_DIR=/tmp/$USER/torch_cache
export TORCHINDUCTOR_CACHE_DIR=/tmp/$USER/torch_cache
export HF_HOME=/tmp/$USER/hf_cache


batch_number=$1
model_path=MODEL_PATH
logs_path=LOGS_PATH
output_path=OUTPUT_PATH

mkdir -p $logs_path
mkdir -p $output_path

echo $logs_path
echo $model_path

# activate environment with vllm installed
source /user/miniconda3/bin/activate vllm

PYTHONUNBUFFERED=1 python ~/int/propose_interventions.py \
    --input_dataset_name CMU-AIRe/InT-SFT \
    --dataset_split train \
    --dataset_start $((batch_number * 1024)) \
    --dataset_end $(((batch_number + 1) * 1024)) \
    --output_path $output_path \
    -K 4 \
    --model $model_path \
    --temperature 0.7 \
    --top_p 0.8 \
    --top_k 20 \
    --max_tokens 16384 \
    --tensor_parallel_size 1 \
    --batch_size 256 > $logs_path/propose_interventions_${model_size}_${batch_number}.log 2>&1
