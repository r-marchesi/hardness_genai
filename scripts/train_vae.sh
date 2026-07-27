#!/bin/bash
#SBATCH --job-name=vae_hardness
#SBATCH --partition=h200
#SBATCH --nodelist=euler
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:3g.71gb:1
#SBATCH --mem=120G
#SBATCH --qos=normal
#SBATCH --container-image=/storage/DSH/projects/hardness_genai/image.sqsh
#SBATCH --container-mounts=/storage/DSH/projects/hardness_genai/,/public_datasets/PublicDatasets/
#SBATCH --output=/storage/DSH/projects/hardness_genai/outputs/vae_%j.out
#SBATCH --error=/storage/DSH/projects/hardness_genai/outputs/vae_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rmarchesi@fbk.eu

echo "Starting Visual-VAE Training on CIFAR-100"
cd /storage/DSH/projects/hardness_genai/repos_genai/PyTorch-VAE


# Launch training
python ../../scripts/train_cvae_cifar100.py

# Launch generation
python ../../scripts/generate_cvae_cifar100.py

echo "Job Complete or Exited!"