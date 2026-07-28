#!/bin/bash
#SBATCH --job-name=vqgan_hardness
#SBATCH --partition=h200
#SBATCH --nodelist=euler
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:3g.71gb:1
#SBATCH --mem=120G
#SBATCH --qos=normal
#SBATCH --container-image=/storage/DSH/projects/hardness_genai/stylegan_image.sqsh
#SBATCH --container-mounts=/storage/DSH/projects/hardness_genai/,/public_datasets/PublicDatasets/
#SBATCH --output=/storage/DSH/projects/hardness_genai/outputs/vqgan_%j.out
#SBATCH --error=/storage/DSH/projects/hardness_genai/outputs/vqgan_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rmarchesi@fbk.eu


PROJECT_ROOT="/storage/DSH/projects/hardness_genai"

# CD into where the python script lives
cd $PROJECT_ROOT/scripts/maskgit/

# Environment variables for PyTorch / CUDA toolchain
export NCCL_P2P_DISABLE=1
export CUDA_HOME=/opt/compiler_env
export PATH=/opt/compiler_env/bin:$PATH
export CPATH=$CUDA_HOME/include:$CUDA_HOME/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

echo "Starting MaskGIT Stage 1 (VQGAN) Training..."

/opt/conda/bin/python train_vqgan.py

echo "VQGAN Job Complete!"