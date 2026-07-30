import os
import torch
from torchvision import datasets
from muse_maskgit_pytorch import VQGanVAE, VQGanVAETrainer

def prepare_cifar_folder(data_dir, dataset_root):
    """Extracts CIFAR-100 into a raw image folder for the VQGAN Trainer."""
    if not os.path.exists(data_dir):
        print(f"Extracting CIFAR-100 images to {data_dir}...")
        os.makedirs(data_dir, exist_ok=True)
        # Download/load dataset using dataset_root
        ds = datasets.CIFAR100(root=dataset_root, train=True, download=True)
        for i, (img, label) in enumerate(ds):
            img.save(os.path.join(data_dir, f"{i:05d}_{label}.jpg"))
        print("Extraction complete!")
    else:
        print(f"Dataset already found at {data_dir}. Skipping extraction.")

def main():
    # Relative to /storage/DSH/projects/hardness_genai/scripts/maskgit/
    dataset_root = "../../data"
    data_dir = "../../data/cifar100_raw_images"
    results_dir = "../../checkpoints/maskgit/vqgan"
    os.makedirs(results_dir, exist_ok=True)

    prepare_cifar_folder(data_dir, dataset_root)

    # 1. Initialize the VQGAN
    vae = VQGanVAE(
        dim = 64,
        channels = 3,
        layers = 2,
        discr_layers = 2,           
        num_tokens = 8192,
        temperature = 0.9
    )

    # 2. Initialize the Trainer
    trainer = VQGanVAETrainer(
        vae = vae,
        image_size = 32,            # CIFAR-100 resolution
        folder = data_dir,          # Point to the extracted images
        num_train_steps = 50000,    # Total optimization steps
        lr = 3e-4,
        batch_size = 128,
        grad_accum_every = 1,
        save_results_every = 1000,  # Save reconstructions to monitor progress
        save_model_every = 1000,    # Save checkpoints
        results_folder = results_dir
    )

    print("Starting VQGAN Stage 1 Training...")
    trainer.train()
    print("Stage 1 Training Complete!")

if __name__ == '__main__':
    main()