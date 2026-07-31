import os
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
from muse_maskgit_pytorch import VQGanVAE

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

def run_diagnostics():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("--- Testing Sanity Check VAE Reconstruction Quality ---")
    
    # Must match the new high-capacity parameters
    vae = VQGanVAE(dim=128, channels=3, layers=2, discr_layers=2, codebook_size=4096, temperature=0.9).to(device)
    
    # Grab the checkpoint from step 4000 or 5000 (adjust if necessary based on what saved)
    ema_path = os.path.join(PROJECT_ROOT, "checkpoints/maskgit/vqgan/vae.4000.ema.pt")
    
    if not os.path.exists(ema_path):
        # Fallback in case 4000 didn't save, try 5000
        ema_path = os.path.join(PROJECT_ROOT, "checkpoints/maskgit/vqgan/vae.5000.ema.pt")
        if not os.path.exists(ema_path):
            print(f"Checkpoint not found at {ema_path}!")
            return

    ema_state_dict = torch.load(ema_path, map_location=device)
    clean_state_dict = {k.replace("ema_model.", ""): v for k, v in ema_state_dict.items() if k.startswith("ema_model.")}
    vae.load_state_dict(clean_state_dict)
    vae.eval()
    
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.CIFAR100(root=os.path.join(PROJECT_ROOT, "data"), train=False, download=True, transform=transform)
    real_imgs = torch.stack([dataset[i][0] for i in range(8)]).to(device)
    
    with torch.no_grad():
        encoded, _, _ = vae.encode(real_imgs)
        recon_imgs = vae.decode(encoded)
        
    comparison = torch.cat([real_imgs, recon_imgs.clamp(0, 1)], dim=0)
    out_img = os.path.join(SCRIPT_DIR, "sanity_reconstruction_test.jpg")
    save_image(comparison, out_img, nrow=8)
    
    print(f">> Saved {out_img}")
    print(">> Open this image. At 5,000 steps, the colors might be noisy, but the EDGES should be sharp.")

if __name__ == "__main__":
    run_diagnostics()