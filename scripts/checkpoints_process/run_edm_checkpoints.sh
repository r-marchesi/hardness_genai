#!/bin/bash
#SBATCH --job-name=edm_checkpoints
#SBATCH --partition=h200
#SBATCH --nodelist=euler
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:3g.71gb:1
#SBATCH --mem=120G
#SBATCH --qos=normal
#SBATCH --container-image=/storage/DSH/projects/hardness_genai/image.sqsh
#SBATCH --container-mounts=/storage/DSH/projects/hardness_genai/,/public_datasets/PublicDatasets/
#SBATCH --output=/storage/DSH/projects/hardness_genai/outputs/edm_checkpoints_%j.out
#SBATCH --error=/storage/DSH/projects/hardness_genai/outputs/edm_checkpoints_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rmarchesi@fbk.eu

PROJECT_ROOT="/storage/DSH/projects/hardness_genai"

export CUDA_HOME=/opt/compiler_env
export PATH=/opt/compiler_env/bin:$PATH
export CPATH=$CUDA_HOME/include:$CUDA_HOME/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

cd $PROJECT_ROOT/scripts/checkpoints_process

# Ensure torch-fidelity is installed
python -m pip install torch-fidelity

echo "Starting EDM Checkpoints Generation and Evaluation..."

python process_edm_checkpoints.py

echo "EDM Checkpoints Processing Complete!"