#!/bin/bash
#SBATCH --job-name=maskgit_hardness
#SBATCH --partition=h200
#SBATCH --nodelist=euler
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:3g.71gb:1
#SBATCH --mem=120G
#SBATCH --qos=normal
#SBATCH --container-image=/storage/DSH/projects/hardness_genai/image.sqsh
#SBATCH --container-mounts=/storage/DSH/projects/hardness_genai/,/public_datasets/PublicDatasets/
#SBATCH --output=/storage/DSH/projects/hardness_genai/outputs/maskgit_%j.out
#SBATCH --error=/storage/DSH/projects/hardness_genai/outputs/maskgit_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rmarchesi@fbk.eu


echo "Starting MaskGIT Training on CIFAR-100"
cd /storage/DSH/projects/hardness_genai/repos_genai/MaskGIT-pytorch

echo "Stage 1: Training VQGAN Tokenizer..."
python training_vqgan.py --dataset cifar100 --batch-size 128

echo "Stage 2: Training Bidirectional Transformer..."
python training_transformer.py --dataset cifar100 --batch-size 128

echo "Job Complete or Exited!"