import os
import sys
import json
import torch
import pickle
from PIL import Image

sys.path.insert(0, os.getcwd())
from generate import edm_sampler

def generate_samples():
    device = torch.device('cuda')
    
    network_pkl = '../../checkpoints/edm/00000-cifar100_custom-cond-ddpmpp-edm-gpus1-batch512-fp32/network-snapshot-097843.pkl'
    
    print(f'Loading EDM network from: {network_pkl}')
    with open(network_pkl, 'rb') as f:
        net = pickle.load(f)['ema'].to(device)
        
    # Define the updated output directory
    outdir = '../../outputs/synthetic_data/cifar100/edm'
    os.makedirs(outdir, exist_ok=True)
    
    # Load CIFAR-100 resampling ratios from the JSON file
    json_path = "../../data/oversampling_targets.json"
    with open(json_path, 'r') as f:
        resampling_ratios = json.load(f)
    
    sample_id = 0
    batch_size = 100  # Smaller batch size for EDM due to diffusion memory constraints
    
    for label, count in enumerate(resampling_ratios):
        # Double the requested amount for stability analysis
        target_count = count * 2
        if target_count == 0:
            continue
            
        print(f"Generating {target_count} samples for class {label}...")
        
        generated_so_far = 0
        while generated_so_far < target_count:
            current_batch = min(batch_size, target_count - generated_so_far)
            
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
                img.save(os.path.join(outdir, f"{sample_id}_{label}.jpg"))
                sample_id += 1
                
            generated_so_far += current_batch
            
    print(f'Successfully saved {sample_id} EDM samples to {outdir}')

if __name__ == '__main__':
    generate_samples()