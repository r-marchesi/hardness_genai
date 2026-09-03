#!/bin/bash
#SBATCH --job-name=resume_stylegan_hardness
#SBATCH --partition=h200
#SBATCH --nodelist=euler
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:3g.71gb:1
#SBATCH --mem=120G
#SBATCH --qos=normal
#SBATCH --container-image=/storage/DSH/projects/hardness_genai/stylegan_image.sqsh
#SBATCH --container-mounts=/storage/DSH/projects/hardness_genai/,/public_datasets/PublicDatasets/
#SBATCH --output=/storage/DSH/projects/hardness_genai/outputs/resume_stylegan_%j.out
#SBATCH --error=/storage/DSH/projects/hardness_genai/outputs/resume_stylegan_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rmarchesi@fbk.eu

PROJECT_ROOT="/storage/DSH/projects/hardness_genai"

export CUDA_HOME=/opt/compiler_env
export PATH=/opt/compiler_env/bin:$PATH
export CPATH=$CUDA_HOME/include:$CUDA_HOME/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

cd $PROJECT_ROOT/repos_genai/stylegan3

echo "Resuming Official Conditional StyleGAN Training..."

RESUME_PKL="$PROJECT_ROOT/checkpoints/stylegan/00025-stylegan2-cifar100_stylegan-gpus1-batch64-gamma0.01/network-snapshot-025000.pkl"

/opt/conda/bin/python train.py \
    --outdir=$PROJECT_ROOT/checkpoints/stylegan \
    --cfg=stylegan2 \
    --data=$PROJECT_ROOT/data/cifar100_stylegan.zip \
    --gpus=1 \
    --batch=64 \
    --gamma=0.01 \
    --cond=1 \
    --mirror=1 \
    --resume=$RESUME_PKL

echo "Training Job Complete!"