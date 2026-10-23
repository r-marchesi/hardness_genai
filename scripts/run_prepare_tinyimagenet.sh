#!/bin/bash
#SBATCH --job-name=prep_tiny
#SBATCH --partition=h200
#SBATCH --nodelist=euler
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --qos=normal
#SBATCH --container-image=/storage/DSH/projects/hardness_genai/stylegan_image.sqsh
#SBATCH --container-mounts=/storage/DSH/projects/hardness_genai/,/public_datasets/PublicDatasets/
#SBATCH --output=/storage/DSH/projects/hardness_genai/outputs/prep_tiny_%j.out
#SBATCH --error=/storage/DSH/projects/hardness_genai/outputs/prep_tiny_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rmarchesi@fbk.eu

PROJECT_ROOT="/storage/DSH/projects/hardness_genai"
cd $PROJECT_ROOT

echo "Starting Tiny-ImageNet Download and Formatting..."

# Execute the preparation script
/opt/conda/bin/python scripts/prepare_tinyimagenet.py

echo "Dataset preparation complete!"