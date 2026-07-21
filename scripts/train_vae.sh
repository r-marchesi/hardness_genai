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
cd /storage/DSH/projects/hardness_genai/repos_genai/visual-vae

# 1. Install TensorFlow locally (required by the repo's internal config)
pip install --user tensorflow

# 2. Create a config for CIFAR-100 from the CIFAR-10 template
cp -n config/cifar10.py config/cifar100.py

# Replace instances of 'cifar10' with 'cifar100' inside the new config file
sed -i 's/cifar10/cifar100/g' config/cifar100.py

# 3. Launch training
python scripts/train.py --config config/cifar100.py --global_dir ../../checkpoints/vae

echo "Job Complete or Exited!"