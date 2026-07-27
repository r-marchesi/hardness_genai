import os
import sys
import torch
import pickle
from PIL import Image

# Ensure Python can locate the custom network classes inside the EDM repository
sys.path.insert(0, os.getcwd())

# Import the official stochastic sampler directly from their script
from generate import edm_sampler

def generate_samples():
    device = torch.device('cuda')
    
    # Point directly to your specific checkpoint
    network_pkl = '../../checkpoints/edm/00000-cifar100_custom-cond-ddpmpp-edm-gpus1-batch512-fp32/network-snapshot-097843.pkl'
    
    print(f'Loading EDM network from: {network_pkl}')
    with open(network_pkl, 'rb') as f:
        net = pickle.load(f)['ema'].to(device)
        
    outdir = '../../outputs/edm_samples'
    os.makedirs(outdir, exist_ok=True)
    
    print('Generating 100 conditionally labeled images (one per CIFAR-100 class)...')
    
    # We will generate them in one large batch (100 images) 
    # An H200 has 141GB of VRAM; a batch of 100 32x32 images is practically nothing.
    batch_size = 100
    
    # 1. Prepare latents (noise) using manual seeds for exact reproducibility
    latents = torch.zeros([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
    for i in range(batch_size):
        torch.manual_seed(i)  # Tie the random seed to the class ID
        latents[i] = torch.randn([net.img_channels, net.img_resolution, net.img_resolution], device=device)
        
    # 2. Prepare one-hot class labels 
    # torch.eye creates a perfect 100x100 identity matrix, meaning row 0 is class 0, row 1 is class 1, etc.
    class_labels = torch.eye(net.label_dim, device=device)
    
    # 3. Generate using their official sampler
    # 18 steps is the recommended baseline for CIFAR in the EDM paper
    with torch.no_grad():
        images = edm_sampler(net, latents, class_labels, num_steps=18)
        
    # 4. Denormalize and save
    # EDM outputs are roughly [-1, 1] which need to be scaled back to standard RGB [0, 255]
    images = (images * 127.5 + 128).clip(0, 255).to(torch.uint8)
    
    # Convert from PyTorch format [B, C, H, W] to Pillow format [B, H, W, C]
    images = images.permute(0, 2, 3, 1).cpu().numpy() 
    
    for i in range(batch_size):
        img = Image.fromarray(images[i], 'RGB')
        img.save(os.path.join(outdir, f'edm_cifar100_class{i:03d}.png'))
        
    print(f'Successfully saved 100 labeled EDM images to {outdir}')

if __name__ == '__main__':
    generate_samples()