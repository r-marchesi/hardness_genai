import os
import sys
import torch
import pickle
import argparse
from PIL import Image
from tqdm import tqdm

# Dynamically resolve PROJECT_ROOT based on script location 
# (Assuming this is saved in /storage/DSH/projects/hardness_genai/scripts/generate_balanced/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

# Add potential locations of 'generate.py' to sys.path to replace os.getcwd()
sys.path.insert(0, os.path.join(PROJECT_ROOT, "repos_genai/edm"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
sys.path.insert(0, PROJECT_ROOT)

from generate import edm_sampler

def generate_balanced_samples(ckpt_path, out_dir, batch_size=100, samples_per_class=500, num_classes=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Generating on: {device}")
    
    print(f'Loading EDM network from: {ckpt_path}')
    with open(ckpt_path, 'rb') as f:
        net = pickle.load(f)['ema'].to(device)
        
    os.makedirs(out_dir, exist_ok=True)
    
    sample_id = 0
    print(f"Generating balanced dataset: {samples_per_class} samples for {num_classes} classes.")
    
    for label in tqdm(range(num_classes), desc="Classes"):
        # Create a subfolder for each class to keep the dataset structured for evaluation
        class_dir = os.path.join(out_dir, str(label))
        os.makedirs(class_dir, exist_ok=True)
        
        generated_so_far = 0
        while generated_so_far < samples_per_class:
            current_batch = min(batch_size, samples_per_class - generated_so_far)
            
            # Prepare latents
            latents = torch.randn([current_batch, net.img_channels, net.img_resolution, net.img_resolution], device=device)
            
            # Prepare one-hot class labels for this specific batch
            class_labels = torch.zeros([current_batch, net.label_dim], device=device)
            class_labels[:, label] = 1.0
            
            with torch.no_grad():
                images = edm_sampler(net, latents, class_labels, num_steps=18)
                
            # Denormalize and convert
            images = (images * 127.5 + 128).clip(0, 255).to(torch.uint8)
            images = images.permute(0, 2, 3, 1).cpu().numpy() 
            
            for i in range(current_batch):
                img = Image.fromarray(images[i], 'RGB')
                img_name = f"img_{generated_so_far + i:04d}.jpg"
                img.save(os.path.join(class_dir, img_name))
                sample_id += 1
                
            generated_so_far += current_batch
            
    print(f'Successfully saved {sample_id} EDM samples to {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate balanced EDM dataset for CIFAR-100")
    
    # Default to the checkpoint used in your oversampling script
    default_ckpt = os.path.join(
        PROJECT_ROOT, 
        "checkpoints/edm/00000-cifar100_custom-cond-ddpmpp-edm-gpus1-batch512-fp32/network-snapshot-097843.pkl"
    )
    default_out = os.path.join(PROJECT_ROOT, "outputs/synthetic_data/cifar100_balanced/edm")
    
    parser.add_argument("--ckpt", type=str, default=default_ckpt, help="Path to EDM checkpoint")
    parser.add_argument("--out_dir", type=str, default=default_out, help="Output directory")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for generation")
    
    args = parser.parse_args()

    generate_balanced_samples(args.ckpt, args.out_dir, args.batch_size)