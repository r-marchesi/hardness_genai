import os
import sys
import json
import torch
import pickle
from PIL import Image

# Dynamically resolve PROJECT_ROOT based on script location (/storage/DSH/projects/hardness_genai)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../"))

# Add StyleGAN3 repository to Python path
STYLEGAN_REPO = os.path.join(PROJECT_ROOT, "repos_genai/stylegan3")
sys.path.insert(0, STYLEGAN_REPO)

import dnnlib
import legacy

def generate_samples():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Generating on: {device}")
    
    # Path to snapshot
    network_pkl = os.path.join(
        PROJECT_ROOT, 
        "checkpoints/stylegan/00016-stylegan2-cifar100_stylegan-gpus1-batch64-gamma0.01/network-snapshot-011289.pkl"
    )
    
    print(f'Loading StyleGAN network from: {network_pkl}')
    with open(network_pkl, 'rb') as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
        
    outdir = os.path.join(PROJECT_ROOT, "outputs/synthetic_data/cifar100/stylegan")
    os.makedirs(outdir, exist_ok=True)
    
    json_path = os.path.join(PROJECT_ROOT, "data/oversampling_targets.json")
    with open(json_path, 'r') as f:
        resampling_ratios = json.load(f)
        
    sample_id = 0
    batch_size = 100 
    
    for label, count in enumerate(resampling_ratios):
        target_count = count * 2
        if target_count == 0:
            continue
            
        print(f"Generating {target_count} samples for class {label}...")
        
        generated_so_far = 0
        while generated_so_far < target_count:
            current_batch = min(batch_size, target_count - generated_so_far)
            
            z = torch.randn([current_batch, G.z_dim], device=device)
            c = torch.zeros([current_batch, G.c_dim], device=device)
            c[:, label] = 1.0
            
            with torch.no_grad():
                img = G(z, c)
                
            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img = img.permute(0, 2, 3, 1).cpu().numpy()
            
            for i in range(current_batch):
                pil_img = Image.fromarray(img[i], 'RGB')
                pil_img.save(os.path.join(outdir, f"{sample_id}_{label}.jpg"))
                sample_id += 1
                
            generated_so_far += current_batch
            
    print(f'Successfully saved {sample_id} StyleGAN samples to {outdir}')

if __name__ == '__main__':
    generate_samples()