import os
import sys
import json
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

# Ensure Python can locate modules in the current working directory
sys.path.insert(0, os.getcwd())
from models.cvae import ConditionalVAE

# Redefine the patched architecture
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

    # Initialize Model & Load Weights
    model = PatchedConditionalVAE(
        in_channels=3, 
        num_classes=100, 
        latent_dim=512, 
        img_size=32, 
        hidden_dims=[64, 128, 256, 512]
    ).to(device)

    weights_path = "../../checkpoints/vae/cvae_cifar100_best.pt"
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Checkpoint not found at {weights_path}.")

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # Define the updated output directory
    out_dir = "../../outputs/synthetic_data/cifar100/vae"
    os.makedirs(out_dir, exist_ok=True)

    # Load CIFAR-100 resampling ratios from the JSON file
    json_path = "../../data/oversampling_targets.json"
    with open(json_path, 'r') as f:
        resampling_ratios = json.load(f)

    sample_id = 0
    batch_size = 500  # Process in chunks to avoid out-of-memory errors

    for label, count in enumerate(resampling_ratios):
        # Double the requested amount for stability analysis
        target_count = count * 2
        if target_count == 0:
            continue
            
        print(f"Generating {target_count} samples for class {label}...")
        
        generated_so_far = 0
        while generated_so_far < target_count:
            current_batch = min(batch_size, target_count - generated_so_far)
            
            # Prepare batch labels
            labels = torch.full((current_batch,), label, dtype=torch.long, device=device)
            labels_onehot = F.one_hot(labels, num_classes=100).float()
            
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
                save_path = os.path.join(out_dir, f"{sample_id}_{label}.jpg")
                save_image(generated_imgs[i], save_path)
                sample_id += 1
                
            generated_so_far += current_batch

    print(f"Successfully saved {sample_id} CVAE samples to {out_dir}")

if __name__ == '__main__':
    generate_samples()