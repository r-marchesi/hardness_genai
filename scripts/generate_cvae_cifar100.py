import os
import sys

# Ensure Python can locate modules in the current working directory
sys.path.insert(0, os.getcwd())

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

# Import the base class from the local repository
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
        latent_dim=128, 
        img_size=32, 
        hidden_dims=[32, 64, 128, 256]
    ).to(device)

    # Point this to your best checkpoint
    weights_path = "../../checkpoints/vae/cvae_cifar100_best.pt"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # ==========================================
    # 3. SETUP GENERATION PARAMETERS
    # ==========================================
    # Let's generate 1 image for every single class in CIFAR-100 (100 images total)
    labels = torch.arange(100).to(device)
    
    # Or, if you wanted to generate 100 images of JUST class '3' (e.g., bear):
    # labels = torch.full((100,), 3).to(device)

    # One-hot encode the labels
    labels_onehot = F.one_hot(labels, num_classes=100).float().to(device)

    # Sample random Gaussian noise (Mean 0, Variance 1)
    # The batch size must match the number of labels we created
    num_samples = labels.size(0)
    z = torch.randn(num_samples, 128).to(device)

    # ==========================================
    # 4. GENERATE & DENORMALIZE
    # ==========================================
    with torch.no_grad():
        # The AntixK decode method expects the noise and labels concatenated together
        z_concat = torch.cat([z, labels_onehot], dim=1)
        generated_imgs = model.decode(z_concat)

    # Denormalize from [-1, 1] back to [0, 1] for saving
    generated_imgs = (generated_imgs * 0.5) + 0.5
    
    # Clamp to ensure no pixel values overshoot due to minor numerical errors
    generated_imgs = generated_imgs.clamp(0, 1)

    # ==========================================
    # 5. SAVE THE OUTPUT
    # ==========================================
    out_dir = "../../outputs/vae_samples"
    os.makedirs(out_dir, exist_ok=True)
    
    # Save as a grid (10x10)
    save_path = os.path.join(out_dir, "cvae_cifar100_grid.png")
    save_image(generated_imgs, save_path, nrow=10)
    
    print(f"Successfully saved 100 generated samples to {save_path}")

if __name__ == '__main__':
    generate_samples()