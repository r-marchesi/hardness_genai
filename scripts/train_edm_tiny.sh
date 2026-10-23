#!/bin/bash
#SBATCH --job-name=edm_tiny
#SBATCH --partition=h200
#SBATCH --nodelist=euler
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:3g.71gb:1
#SBATCH --mem=120G
#SBATCH --qos=normal
#SBATCH --container-image=/storage/DSH/projects/hardness_genai/image.sqsh
#SBATCH --container-mounts=/storage/DSH/projects/hardness_genai/,/public_datasets/PublicDatasets/
#SBATCH --output=/storage/DSH/projects/hardness_genai/outputs/edm_tiny_%j.out
#SBATCH --error=/storage/DSH/projects/hardness_genai/outputs/edm_tiny_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rmarchesi@fbk.eu

PROJECT_ROOT="/storage/DSH/projects/hardness_genai"

# Isolate shared memory files to your permanent storage immediately
mkdir -p $PROJECT_ROOT/tmp
export TMPDIR=$PROJECT_ROOT/tmp

echo "Starting EDM Training on Tiny-ImageNet (Single GPU MIG)"
cd /storage/DSH/projects/hardness_genai/repos_genai/edm

# Launching on 1 GPU with Gradient Accumulation
torchrun --standalone --nproc_per_node=1 train.py \
    --outdir=../../checkpoints/edm_tiny \
    --data=../../data/tinyimagenet_custom.zip \
    --cond=1 \
    --arch=ddpmpp \
    --duration=1000 \
    --batch=512 \
    --batch-gpu=64

echo "Job Complete or Exited!"