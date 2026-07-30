import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1" 

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from muse_maskgit_pytorch import VQGanVAE, MaskGit, MaskGitTransformer

def run_diagnostics():
    print("--- STARTING MASKGIT DIAGNOSTICS ---")
    
    print("\n1. Loading VAE...")
    vae = VQGanVAE(dim=64, channels=3, layers=2, discr_layers=2, num_tokens=8192, temperature=0.9).cuda()
    ema_state_dict = torch.load("../../checkpoints/maskgit/vqgan/vae.49000.ema.pt", map_location="cuda")
    
    clean_state_dict = {k.replace("ema_model.", ""): v for k, v in ema_state_dict.items() if k.startswith("ema_model.")}
    vae.load_state_dict(clean_state_dict)
    vae.eval()
    
    print("\n2. Loading test batch...")
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.CIFAR100(root="../../data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    images, _ = next(iter(dataloader))
    images = images.cuda()
    
    print(f"-> Input Images Shape: {images.shape}")
    
    print("\n3. Testing VAE Token Output...")
    with torch.no_grad():
        # The correct method in muse-maskgit-pytorch
        _, token_indices, _ = vae.encode(images)
        
    print(f"-> Output Grid Shape: {token_indices.shape}")
    print(f"-> Min Token ID: {token_indices.min().item()}")
    print(f"-> Max Token ID: {token_indices.max().item()}")
    
    if token_indices.max().item() >= 8192:
        print("\n!!! CRITICAL FINDING !!!")
        print("Your VAE is outputting tokens larger than the 8192 codebook size.")
        print("This means the Transformer's output layer is too small, causing the NLLLoss2d crash!")
        return

    print("\n4. Building Transformer...")
    seq_len = token_indices.shape[1]
    print(f"-> Using True Sequence Length: {seq_len}")
    
    transformer = MaskGitTransformer(
        num_tokens = 8192,
        seq_len = seq_len, 
        dim = 512, depth = 6, dim_head = 64, heads = 8, ff_mult = 4, flash = False
    ).cuda()
    
    transformer.num_tokens = vae.codebook_size
    transformer.token_emb = torch.nn.Embedding(vae.codebook_size + 1, 512).cuda()
    
    print("\n5. Running MaskGIT Wrapper...")
    maskgit = MaskGit(vae=vae, transformer=transformer, image_size=32, cond_drop_prob=0.1).cuda()
    
    try:
        loss = maskgit(images, texts=["class 0", "class 1", "class 2", "class 3"])
        print(f"\nSUCCESS! Forward pass completed. Loss: {loss.item()}")
    except Exception as e:
        print(f"\n!!! FORWARD PASS CRASHED !!!")
        print(e)
        
if __name__ == "__main__":
    run_diagnostics()
