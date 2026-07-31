import os
import sys
import torch
import argparse
from PIL import Image
from tqdm import tqdm

# Dynamically resolve PROJECT_ROOT based on script location 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# FIX: Go up TWO levels (from scripts/generate_balanced to hardness_genai)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

# Add StyleGAN3 repository to Python path
STYLEGAN_REPO = os.path.join(PROJECT_ROOT, "repos_genai/stylegan3")
sys.path.insert(0, STYLEGAN_REPO)

import dnnlib
import legacy


def generate_balanced_samples(ckpt_path, out_dir, batch_size=100, samples_per_class=500, num_classes=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Generating on: {device}")
    
    print(f'Loading StyleGAN network from: {ckpt_path}')
    with open(ckpt_path, 'rb') as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
        
    os.makedirs(out_dir, exist_ok=True)
    
    sample_id = 0
    print(f"Generating balanced dataset: {samples_per_class} samples for {num_classes} classes.")
    
    for label in tqdm(range(num_classes), desc="Classes"):
        # Create a subfolder for each class to keep the dataset structured
        class_dir = os.path.join(out_dir, str(label))
        os.makedirs(class_dir, exist_ok=True)
        
        generated_so_far = 0
        while generated_so_far < samples_per_class:
            current_batch = min(batch_size, samples_per_class - generated_so_far)
            
            # Generate latents and one-hot condition vectors
            z = torch.randn([current_batch, G.z_dim], device=device)
            c = torch.zeros([current_batch, G.c_dim], device=device)
            c[:, label] = 1.0
            
            with torch.no_grad():
                img = G(z, c)
                
            # Post-process the output tensor into a standard image format
            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img = img.permute(0, 2, 3, 1).cpu().numpy()
            
            for i in range(current_batch):
                pil_img = Image.fromarray(img[i], 'RGB')
                img_name = f"img_{generated_so_far + i:04d}.jpg"
                pil_img.save(os.path.join(class_dir, img_name))
                sample_id += 1
                
            generated_so_far += current_batch
            
    print(f'Successfully saved {sample_id} StyleGAN samples to {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate balanced StyleGAN dataset for CIFAR-100")
    
    # Default to the checkpoint used in your oversampling script
    default_ckpt = os.path.join(
        PROJECT_ROOT, 
        "checkpoints/stylegan/00016-stylegan2-cifar100_stylegan-gpus1-batch64-gamma0.01/network-snapshot-011289.pkl"
    )
    default_out = os.path.join(PROJECT_ROOT, "outputs/synthetic_data/cifar100_balanced/stylegan")
    
    parser.add_argument("--ckpt", type=str, default=default_ckpt, help="Path to StyleGAN checkpoint")
    parser.add_argument("--out_dir", type=str, default=default_out, help="Output directory")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for generation")
    
    args = parser.parse_args()

    generate_balanced_samples(args.ckpt, args.out_dir, args.batch_size)