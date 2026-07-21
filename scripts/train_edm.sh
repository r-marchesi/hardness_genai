#!/bin/bash
#SBATCH --job-name=edm_hardness
#SBATCH --partition=h200
#SBATCH --nodelist=euler
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:3g.71gb:1
#SBATCH --mem=120G
#SBATCH --qos=normal
#SBATCH --container-image=/storage/DSH/projects/hardness_genai/image.sqsh
#SBATCH --container-mounts=/storage/DSH/projects/hardness_genai/,/public_datasets/PublicDatasets/
#SBATCH --output=/storage/DSH/projects/hardness_genai/outputs/edm_%j.out
#SBATCH --error=/storage/DSH/projects/hardness_genai/outputs/edm_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rmarchesi@fbk.eu

echo "Starting EDM Training on CIFAR-100 (Single GPU MIG)"
cd /storage/DSH/projects/hardness_genai/repos_genai/edm

# Launching on 1 GPU
torchrun --standalone --nproc_per_node=1 train.py \
    --outdir=../../checkpoints/edm \
    --data=../../data/cifar100_custom.zip \
    --cond=1 \
    --arch=ddpmpp

echo "Job Complete or Exited!"