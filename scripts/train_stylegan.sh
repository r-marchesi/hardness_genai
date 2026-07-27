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

# Define absolute paths 
PROJECT_ROOT="/storage/DSH/projects/hardness_genai"
COMPILER_ENV="$PROJECT_ROOT/compiler_env_v2"
COMPILER_BIN="$PROJECT_ROOT/compiler_bin_v2"
PYTHON_PACKAGES="$PROJECT_ROOT/python_packages"

cd $PROJECT_ROOT/repos_genai/stylegan3

echo "Setting up dependencies in mounted storage..."

# 1. Create a local package directory in the mounted storage
mkdir -p $PYTHON_PACKAGES

# 2. Use the exact container Python to install dependencies
/opt/conda/bin/python -m pip install --target=$PYTHON_PACKAGES ninja click requests tqdm scipy

# 3. Force Python to look in this specific folder for 'click'
export PYTHONPATH=$PYTHON_PACKAGES:$PYTHONPATH

# 4. Check/Link the C++ toolchain (Skips download if it already exists)
if [ ! -d "$COMPILER_ENV" ]; then
    echo "Downloading toolchain... (This will take ~2 minutes)"
    /opt/conda/bin/conda create -y -p $COMPILER_ENV -c conda-forge -c nvidia \
        gxx_linux-64=11 gcc_linux-64=11 cuda-toolkit=12.1
fi

mkdir -p $COMPILER_BIN
ln -sf $COMPILER_ENV/bin/x86_64-conda-linux-gnu-c++ $COMPILER_BIN/c++
ln -sf $COMPILER_ENV/bin/x86_64-conda-linux-gnu-g++ $COMPILER_BIN/g++
ln -sf $COMPILER_ENV/bin/x86_64-conda-linux-gnu-gcc $COMPILER_BIN/gcc

# 5. Export critical environment variables for PyTorch compilation
export PATH=$COMPILER_BIN:$COMPILER_ENV/bin:$HOME/.local/bin:$PATH
export CXX=c++
export CC=gcc
export CUDA_HOME=$COMPILER_ENV

export CPATH=$CUDA_HOME/include:$CUDA_HOME/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

rm -rf ~/.cache/torch_extensions/

echo "Using C++ compiler at: $(which c++)"
echo "Using CUDA_HOME at: $CUDA_HOME"
echo "Using NVCC at: $(which nvcc)"

echo "Starting Official Conditional StyleGAN Training..."

# 6. Force the use of the container's Python binary to avoid Conda's empty Python
/opt/conda/bin/python train.py \
    --outdir=$PROJECT_ROOT/checkpoints/stylegan \
    --cfg=stylegan2 \
    --data=$PROJECT_ROOT/data/cifar100_stylegan.zip \
    --gpus=1 \
    --batch=64 \
    --gamma=0.01 \
    --cond=1 \
    --mirror=1

echo "Training Job Complete!"