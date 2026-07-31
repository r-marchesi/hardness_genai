import os
import sys
import torch
import argparse
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm

# Dynamically resolve PROJECT_ROOT based on script location 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

# FIX: Add the specific PyTorch-VAE repository to sys.path
VAE_REPO = os.path.join(PROJECT_ROOT, "repos_genai/PyTorch-VAE")
sys.path.insert(0, VAE_REPO)

from models.cvae import ConditionalVAE


# Redefine the patched architecture (retained from original script)
class PatchedConditionalVAE(ConditionalVAE):
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        result = self.decoder_input(z)
        bottleneck_channels = self.decoder_input.out_features // 4
        result = result.view(-1, bottleneck_channels, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

def generate_balanced_samples(ckpt_path, out_dir, batch_size=500, samples_per_class=500, num_classes=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating on: {device}")

    # Initialize Model & Load Weights
    model = PatchedConditionalVAE(
        in_channels=3, 
        num_classes=num_classes, 
        latent_dim=512, 
        img_size=32, 
        hidden_dims=[64, 128, 256, 512]
    ).to(device)

    print(f"Loading VAE checkpoint from: {ckpt_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}.")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    os.makedirs(out_dir, exist_ok=True)
    
    sample_id = 0
    print(f"Generating balanced dataset: {samples_per_class} samples for {num_classes} classes.")

    for label in tqdm(range(num_classes), desc="Classes"):
        # Create a subfolder for each class
        class_dir = os.path.join(out_dir, str(label))
        os.makedirs(class_dir, exist_ok=True)
        
        generated_so_far = 0
        while generated_so_far < samples_per_class:
            current_batch = min(batch_size, samples_per_class - generated_so_far)
            
            # Prepare batch labels
            labels = torch.full((current_batch,), label, dtype=torch.long, device=device)
            labels_onehot = F.one_hot(labels, num_classes=num_classes).float()
            
            # Sample random Gaussian noise
            z = torch.randn(current_batch, 512).to(device)
            
            with torch.no_grad():
                z_concat = torch.cat([z, labels_onehot], dim=1)
                generated_imgs = model.decode(z_concat)
                
            # Denormalize from [-1, 1] back to [0, 1]
            generated_imgs = (generated_imgs * 0.5) + 0.5
            generated_imgs = generated_imgs.clamp(0, 1)
            
            # Save individual outputs
            for i in range(current_batch):
                img_name = f"img_{generated_so_far + i:04d}.jpg"
                save_path = os.path.join(class_dir, img_name)
                save_image(generated_imgs[i], save_path)
                sample_id += 1
                
            generated_so_far += current_batch

    print(f"Successfully saved {sample_id} CVAE samples to {out_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate balanced CVAE dataset for CIFAR-100")
    
    # Default to the checkpoint used in your oversampling script
    default_ckpt = os.path.join(PROJECT_ROOT, "checkpoints/vae/cvae_cifar100_best.pt")
    default_out = os.path.join(PROJECT_ROOT, "outputs/synthetic_data/cifar100_balanced/vae")
    
    parser.add_argument("--ckpt", type=str, default=default_ckpt, help="Path to VAE checkpoint")
    parser.add_argument("--out_dir", type=str, default=default_out, help="Output directory")
    parser.add_argument("--batch_size", type=int, default=500, help="Batch size for generation (VAE can handle larger batches)")
    
    args = parser.parse_args()

    generate_balanced_samples(args.ckpt, args.out_dir, args.batch_size)