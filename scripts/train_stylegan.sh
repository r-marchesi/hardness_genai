#!/bin/bash
#SBATCH --job-name=stylegan_hardness
#SBATCH --partition=h200
#SBATCH --nodelist=euler
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:3g.71gb:1
#SBATCH --mem=120G
#SBATCH --qos=normal
#SBATCH --container-image=/storage/DSH/projects/hardness_genai/image.sqsh
#SBATCH --container-mounts=/storage/DSH/projects/hardness_genai/,/public_datasets/PublicDatasets/
#SBATCH --output=/storage/DSH/projects/hardness_genai/outputs/stylegan_%j.out
#SBATCH --error=/storage/DSH/projects/hardness_genai/outputs/stylegan_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rmarchesi@fbk.eu

echo "Starting StyleGAN2-ADA Training on CIFAR-100"
cd /storage/DSH/projects/hardness_genai/repos_genai/stylegan2-ada-pytorch


# 2. Install the missing C++ build tool into your local user directory
pip install --user ninja

# 3. Launch training
torchrun --standalone --nproc_per_node=1 train.py \
    --outdir=../../checkpoints/stylegan \
    --data=../../data/cifar100_custom.zip \
    --cond=1 \
    --cfg=cifar

echo "Job Complete or Exited!"