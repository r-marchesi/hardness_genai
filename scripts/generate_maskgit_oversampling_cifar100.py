import os
import sys
import json
import torch
from torchvision.utils import save_image

# Monkey-patch HuggingFace Transformers v5 compatibility (Required for text conditioning)
from transformers import PreTrainedTokenizerBase
PreTrainedTokenizerBase.batch_encode_plus = PreTrainedTokenizerBase.__call__

from muse_maskgit_pytorch import VQGanVAE, MaskGit, MaskGitTransformer

# Dynamically resolve PROJECT_ROOT
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../"))

def generate_samples():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating on: {device}")

    # 1. Instantiate the VAE (Exact same parameters as training)
    vae = VQGanVAE(dim=64, channels=3, layers=2, discr_layers=2, num_tokens=65536, temperature=0.9)
    
    # 2. Instantiate the Transformer
    transformer = MaskGitTransformer(
        num_tokens = 65536,
        seq_len = 64,
        dim = 512,
        depth = 6,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        flash = False
    )
    
    # 3. Instantiate MaskGIT
    maskgit = MaskGit(
        vae = vae,
        transformer = transformer,
        image_size = 32,
        cond_drop_prob = 0.1,  
    ).to(device)

    # 4. Load the trained Stage 2 Transformer Weights
    # Assuming epoch 250 is your final checkpoint. Adjust if you stopped earlier.
    weights_path = os.path.join(PROJECT_ROOT, "checkpoints/maskgit/transformer/maskgit_epoch_250.pt")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Checkpoint not found at {weights_path}.")

    print(f"Loading MaskGIT checkpoint from: {weights_path}")
    maskgit.load_state_dict(torch.load(weights_path, map_location=device))
    maskgit.eval()

    # 5. Define output directory
    out_dir = os.path.join(PROJECT_ROOT, "outputs/synthetic_data/cifar100/maskgit")
    os.makedirs(out_dir, exist_ok=True)

    # 6. Load CIFAR-100 resampling ratios
    json_path = os.path.join(PROJECT_ROOT, "data/oversampling_targets.json")
    with open(json_path, 'r') as f:
        resampling_ratios = json.load(f)

    sample_id = 0
    batch_size = 100  # MaskGIT requires iterative decoding, so keep batch size reasonable to avoid OOM

    for label, count in enumerate(resampling_ratios):
        target_count = count * 2
        if target_count == 0:
            continue
            
        print(f"Generating {target_count} samples for class {label}...")
        
        generated_so_far = 0
        while generated_so_far < target_count:
            current_batch = min(batch_size, target_count - generated_so_far)
            
            # MaskGIT was trained with string conditioning ("class 0", "class 1", etc.)
            texts = [f"class {label}" for _ in range(current_batch)]
            
            with torch.no_grad():
                # Generate images. cond_scale > 1.0 applies Classifier-Free Guidance. 
                # (3.0 is a standard starting point for high fidelity)
                generated_imgs = maskgit.generate(texts=texts, cond_scale=3.0)
                
            # Your training script used transforms.ToTensor() which puts images in [0, 1].
            # So the VAE output should already be in [0, 1]. We just clamp it for safety.
            generated_imgs = generated_imgs.clamp(0, 1)
            
            # Save individual outputs
            for i in range(current_batch):
                save_path = os.path.join(out_dir, f"{sample_id}_{label}.jpg")
                save_image(generated_imgs[i], save_path)
                sample_id += 1
                
            generated_so_far += current_batch

    print(f"Successfully saved {sample_id} MaskGIT samples to {out_dir}")

if __name__ == '__main__':
    generate_samples()