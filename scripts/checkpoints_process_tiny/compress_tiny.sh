#!/bin/bash
#SBATCH --job-name=zip_tiny
#SBATCH --partition=h200
#SBATCH --nodelist=euler
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --qos=normal
#SBATCH --container-image=/storage/DSH/projects/hardness_genai/stylegan_image.sqsh
#SBATCH --container-mounts=/storage/DSH/projects/hardness_genai/,/public_datasets/PublicDatasets/
#SBATCH --output=/storage/DSH/projects/hardness_genai/outputs/zip_tiny_%j.out
#SBATCH --error=/storage/DSH/projects/hardness_genai/outputs/zip_tiny_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rmarchesi@fbk.eu

cd /storage/DSH/projects/hardness_genai/outputs/synthetic_data

echo "Starting Tiny-ImageNet ZIP File Compression using Python..."

/opt/conda/bin/python -c '
import os
import zipfile

source_dir = "tinyimagenet_checkpoints"
zip_name = "tinyimagenet_checkpoints.zip"

with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            file_path = os.path.join(root, file)
            zipf.write(file_path, arcname=file_path)
'

echo "ZIP File Compression Complete!"