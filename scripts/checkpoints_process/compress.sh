#!/bin/bash
#SBATCH --job-name=zip
#SBATCH --partition=h200
#SBATCH --nodelist=euler
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:3g.71gb:1
#SBATCH --mem=120G
#SBATCH --qos=normal
#SBATCH --container-image=/storage/DSH/projects/hardness_genai/stylegan_image.sqsh
#SBATCH --container-mounts=/storage/DSH/projects/hardness_genai/,/public_datasets/PublicDatasets/
#SBATCH --output=/storage/DSH/projects/hardness_genai/outputs/zip%j.out
#SBATCH --error=/storage/DSH/projects/hardness_genai/outputs/zip%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rmarchesi@fbk.eu

# Navigate to the synthetic_data directory
cd /storage/DSH/projects/hardness_genai/outputs/synthetic_data

echo "Starting ZIP File Compression using Python..."

# Run an inline Python script to zip the folder and exclude 'oversampling'
/opt/conda/bin/python -c '
import os
import zipfile

source_dir = "cifar100_checkpoints"
zip_name = "cifar100_checkpoints.zip"

with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(source_dir):
        # Prevent os.walk from even entering the oversampling directories
        if "oversampling" in dirs:
            dirs.remove("oversampling")
            
        for file in files:
            file_path = os.path.join(root, file)
            zipf.write(file_path, arcname=file_path)
'

echo "ZIP File Compression Complete!"