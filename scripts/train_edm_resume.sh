#!/bin/bash
#SBATCH --job-name=edm_hardness_resume
#SBATCH --partition=h200
#SBATCH --nodelist=euler
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:3g.71gb:1
#SBATCH --mem=120G
#SBATCH --qos=normal
#SBATCH --container-image=/storage/DSH/projects/hardness_genai/image.sqsh
#SBATCH --container-mounts=/storage/DSH/projects/hardness_genai/,/public_datasets/PublicDatasets/
#SBATCH --output=/storage/DSH/projects/hardness_genai/outputs/resume_edm_%j.out
#SBATCH --error=/storage/DSH/projects/hardness_genai/outputs/resume_edm_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rmarchesi@fbk.eu

PROJECT_ROOT="/storage/DSH/projects/hardness_genai"

# Isolate shared memory files to your permanent storage
mkdir -p $PROJECT_ROOT/tmp
export TMPDIR=$PROJECT_ROOT/tmp

echo "Resuming EDM Training on CIFAR-100 (Single GPU MIG)"
cd /storage/DSH/projects/hardness_genai/repos_genai/edm

# Point directly to the training-state file
RESUME_PT="../../checkpoints/edm/00002-cifar100_custom-cond-ddpmpp-edm-gpus1-batch512-fp32/training-state-200000.pt"

# Launching on 1 GPU
torchrun --standalone --nproc_per_node=1 train.py \
    --outdir=../../checkpoints/edm \
    --data=../../data/cifar100_custom.zip \
    --cond=1 \
    --arch=ddpmpp \
    --duration=1000 \
    --resume=$RESUME_PT

echo "Job Complete or Exited!"