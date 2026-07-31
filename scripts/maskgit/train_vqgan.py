import os
import torch
from torchvision import datasets
from muse_maskgit_pytorch import VQGanVAE, VQGanVAETrainer

def prepare_cifar_folder(data_dir, dataset_root):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        ds = datasets.CIFAR100(root=dataset_root, train=True, download=True)
        for i, (img, label) in enumerate(ds):
            img.save(os.path.join(data_dir, f"{i:05d}_{label}.jpg"))
    else:
        print(f"Dataset already found at {data_dir}. Skipping extraction.")

def main():
    dataset_root = "../../data"
    data_dir = "../../data/cifar100_raw_images"
    results_dir = "../../checkpoints/maskgit/vqgan"
    os.makedirs(results_dir, exist_ok=True)

    prepare_cifar_folder(data_dir, dataset_root)

    vae = VQGanVAE(
        dim = 128,          
        channels = 3,
        layers = 2,
        discr_layers = 1,       
        codebook_size = 256     
    )

    trainer = VQGanVAETrainer(
        vae = vae,
        image_size = 32,            
        folder = data_dir,          
        num_train_steps = 50000,    # THE FIX: Back to the full 50,000 steps!
        lr = 1e-4,                  
        batch_size = 128,
        grad_accum_every = 1,
        save_results_every = 1000,  
        save_model_every = 1000,    
        results_folder = results_dir
    )

    print("Starting VQGAN Stage 1 Training...")
    trainer.train()

if __name__ == '__main__':
    main()