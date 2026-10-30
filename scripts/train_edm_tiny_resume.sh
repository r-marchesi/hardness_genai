#!/bin/bash
#SBATCH --job-name=edm_tiny_resume
#SBATCH --partition=h200
#SBATCH --nodelist=euler
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:3g.71gb:1
#SBATCH --mem=120G
#SBATCH --qos=normal
#SBATCH --container-image=/storage/DSH/projects/hardness_genai/image.sqsh
#SBATCH --container-mounts=/storage/DSH/projects/hardness_genai/,/public_datasets/PublicDatasets/
#SBATCH --output=/storage/DSH/projects/hardness_genai/outputs/resume_edm_tiny_%j.out
#SBATCH --error=/storage/DSH/projects/hardness_genai/outputs/resume_edm_tiny_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rmarchesi@fbk.eu

set -euo pipefail

PROJECT_ROOT="/storage/DSH/projects/hardness_genai"

# Isolate shared memory files
mkdir -p "$PROJECT_ROOT/tmp"
export TMPDIR="$PROJECT_ROOT/tmp"

echo "Resuming EDM Training on Tiny-ImageNet (Single GPU MIG)"

cd "$PROJECT_ROOT/repos_genai/edm"

# Output directory from the original Tiny-ImageNet training
OUTDIR="../../checkpoints/edm_tiny"

# Automatically find the most recently saved training state
mapfile -t STATE_FILES < <(
    find "$OUTDIR" -type f -name 'training-state-*.pt' \
    -printf '%T@ %p\n' | sort -nr
)

# Check that a valid checkpoint exists
if [ ${#STATE_FILES[@]} -eq 0 ]; then
    echo "ERROR: No training-state checkpoint found in $OUTDIR"
    exit 1
fi

RESUME_PT="${STATE_FILES[0]#* }"

echo "Resuming from: $RESUME_PT"

# Launching on 1 GPU with gradient accumulation
torchrun --standalone --nproc_per_node=1 train.py \
    --outdir="$OUTDIR" \
    --data=../../data/tinyimagenet_custom.zip \
    --cond=1 \
    --arch=ddpmpp \
    --duration=1000 \
    --batch=512 \
    --batch-gpu=64 \
    --resume="$RESUME_PT"

echo "Job Complete or Exited!"
