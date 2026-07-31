import os
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image

# Monkey-patch HuggingFace Transformers v5 compatibility
from transformers import PreTrainedTokenizerBase
PreTrainedTokenizerBase.batch_encode_plus = PreTrainedTokenizerBase.__call__

from muse_maskgit_pytorch import VQGanVAE, MaskGit, MaskGitTransformer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

def run_diagnostics():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("--- 1. Testing Stage 1 VAE Reconstruction Quality ---")
    vae = VQGanVAE(dim=64, channels=3, layers=2, discr_layers=2, codebook_size=1024, temperature=0.9).to(device)
    ema_path = os.path.join(PROJECT_ROOT, "checkpoints/maskgit/vqgan/vae.49000.ema.pt")
    
    if not os.path.exists(ema_path):
        print("VAE checkpoint not found!")
        return

    ema_state_dict = torch.load(ema_path, map_location=device)
    clean_state_dict = {k.replace("ema_model.", ""): v for k, v in ema_state_dict.items() if k.startswith("ema_model.")}
    vae.load_state_dict(clean_state_dict)
    vae.eval()
    
    # Load 8 real CIFAR-100 images
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.CIFAR100(root=os.path.join(PROJECT_ROOT, "data"), train=False, download=True, transform=transform)
    real_imgs = torch.stack([dataset[i][0] for i in range(8)]).to(device)
    
    with torch.no_grad():
        # Encode and decode the real images
        encoded, _, _ = vae.encode(real_imgs)
        recon_imgs = vae.decode(encoded)
        
    # Save a comparison image (Top row: Real, Bottom row: VAE Reconstruction)
    comparison = torch.cat([real_imgs, recon_imgs.clamp(0, 1)], dim=0)
    out_img = os.path.join(SCRIPT_DIR, "vae_reconstruction_test.jpg")
    save_image(comparison, out_img, nrow=8)
    
    print(f">> Saved {out_img}")
    print(">> Action Required: Open this image. If the bottom row is blurry, the VAE is underpowered and needs fixing.")

    print("\n--- 2. Testing Stage 2 Transformer Determinism ---")
    transformer = MaskGitTransformer(num_tokens=1024, seq_len=64, dim=512, depth=6, dim_head=64, heads=8, ff_mult=4, flash=False).to(device)
    maskgit = MaskGit(vae=vae, transformer=transformer, image_size=32).to(device)
    trans_path = os.path.join(PROJECT_ROOT, "checkpoints/maskgit/transformer/maskgit_epoch_250.pt")

    if not os.path.exists(trans_path):
        print("Transformer checkpoint not found!")
        return

    maskgit.load_state_dict(torch.load(trans_path, map_location=device))
    maskgit.eval()
    
    with torch.no_grad():
        print("Generating 5 independent images for 'class 0'...")
        # Generate 5 images simultaneously
        texts = ["class 0"] * 5
        generated_imgs = maskgit.generate(texts=texts, cond_scale=1.5)
        
    # Check if the images are mathematically identical
    diff_1_2 = torch.abs(generated_imgs[0] - generated_imgs[1]).mean().item()
    diff_1_5 = torch.abs(generated_imgs[0] - generated_imgs[4]).mean().item()
    
    print(f"Visual difference between Sample 1 and 2: {diff_1_2:.6f}")
    print(f"Visual difference between Sample 1 and 5: {diff_1_5:.6f}")
    
    if diff_1_2 < 0.01:
        print(">> VERDICT: The Transformer is suffering from Greedy Decoding (Mode Collapse).")
        print(">> It is ignoring generation randomness and outputting the exact same tokens.")
    else:
        print(">> VERDICT: The Transformer is healthy and generating diverse samples.")

if __name__ == "__main__":
    run_diagnostics()