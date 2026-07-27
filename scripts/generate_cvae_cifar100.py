import os
import sys
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

# Ensure Python can locate modules in the current working directory (PyTorch-VAE)
sys.path.insert(0, os.getcwd())

# Import the base class from local repository
from models.cvae import ConditionalVAE

# ==========================================
# 1. REDEFINE THE PATCHED ARCHITECTURE
# ==========================================
class PatchedConditionalVAE(ConditionalVAE):
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        result = self.decoder_input(z)
        bottleneck_channels = self.decoder_input.out_features // 4
        result = result.view(-1, bottleneck_channels, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

def generate_samples():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating on: {device}")

    # ==========================================
    # 2. INITIALIZE MODEL & LOAD WEIGHTS
    # ==========================================
    model = PatchedConditionalVAE(
        in_channels=3, 
        num_classes=100, 
        latent_dim=512, 
        img_size=32, 
        hidden_dims=[64, 128, 256, 512]
    ).to(device)

    weights_path = "../../checkpoints/vae/cvae_cifar100_best.pt"
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Checkpoint not found at {weights_path}. Make sure training has completed!")

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # ==========================================
    # 3. SETUP GENERATION PARAMETERS
    # ==========================================
    # Generate 1 image for every class in CIFAR-100 (100 images total)
    labels = torch.arange(100).to(device)
    labels_onehot = F.one_hot(labels, num_classes=100).float().to(device)

    # Sample random Gaussian noise matching latent_dim = 512
    num_samples = labels.size(0)
    
    # Optional: Set a manual seed for reproducible noise across runs
    torch.manual_seed(42)
    z = torch.randn(num_samples, 512).to(device)

    # ==========================================
    # 4. GENERATE & DENORMALIZE
    # ==========================================
    with torch.no_grad():
        z_concat = torch.cat([z, labels_onehot], dim=1)
        generated_imgs = model.decode(z_concat)

    # Denormalize from [-1, 1] back to [0, 1] for image saving
    generated_imgs = (generated_imgs * 0.5) + 0.5
    generated_imgs = generated_imgs.clamp(0, 1)

    # ==========================================
    # 5. SAVE INDIVIDUAL OUTPUTS
    # ==========================================
    out_dir = "../../outputs/vae_samples"
    os.makedirs(out_dir, exist_ok=True)
    
    print('Saving 100 labeled CVAE images (one per CIFAR-100 class)...')
    
    # Loop through the generated batch and save each image separately
    for i in range(num_samples):
        save_path = os.path.join(out_dir, f"cvae_cifar100_class{i:03d}.png")
        save_image(generated_imgs[i], save_path)
    
    print(f"Successfully saved 100 individual samples to {out_dir}")

if __name__ == '__main__':
    generate_samples()