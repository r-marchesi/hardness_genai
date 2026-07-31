import os
import sys
import json
import torch
from torchvision.utils import save_image

# Monkey-patch HuggingFace Transformers v5 compatibility
from transformers import PreTrainedTokenizerBase
PreTrainedTokenizerBase.batch_encode_plus = PreTrainedTokenizerBase.__call__

from muse_maskgit_pytorch import VQGanVAE, MaskGit, MaskGitTransformer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../"))

def generate_samples():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating on: {device}")

    # THE FIX: Match the new healthy codebook size (1024)
    vae = VQGanVAE(dim=64, channels=3, layers=2, discr_layers=2, codebook_size=1024, temperature=0.9)
    
    # THE FIX: Match the new healthy token count (1024)
    transformer = MaskGitTransformer(
        num_tokens = 1024,
        seq_len = 64,
        dim = 512,
        depth = 6,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        flash = False
    )
    
    maskgit = MaskGit(
        vae = vae,
        transformer = transformer,
        image_size = 32,
        cond_drop_prob = 0.1,  
    ).to(device)

    # Load your new Stage 2 Transformer Weights
    weights_path = os.path.join(PROJECT_ROOT, "checkpoints/maskgit/transformer/maskgit_epoch_250.pt")
    print(f"Loading MaskGIT checkpoint from: {weights_path}")
    maskgit.load_state_dict(torch.load(weights_path, map_location=device))
    maskgit.eval()

    out_dir = os.path.join(PROJECT_ROOT, "outputs/synthetic_data/cifar100/maskgit")
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(PROJECT_ROOT, "data/oversampling_targets.json")
    with open(json_path, 'r') as f:
        resampling_ratios = json.load(f)

    sample_id = 0
    batch_size = 100 

    for label, count in enumerate(resampling_ratios):
        target_count = count * 2
        if target_count == 0:
            continue
            
        print(f"Generating {target_count} samples for class {label}...")
        
        generated_so_far = 0
        while generated_so_far < target_count:
            current_batch = min(batch_size, target_count - generated_so_far)
            texts = [f"class {label}" for _ in range(current_batch)]
            
            with torch.no_grad():
                # cond_scale=1.5 gives a good balance of class adherence and token diversity
                generated_imgs = maskgit.generate(texts=texts, cond_scale=1.5)
                
            # The VAE naturally outputs in the [0, 1] range. Clamp just in case.
            generated_imgs = generated_imgs.clamp(0, 1)
            
            for i in range(current_batch):
                save_path = os.path.join(out_dir, f"{sample_id}_{label}.jpg")
                save_image(generated_imgs[i], save_path)
                sample_id += 1
                
            generated_so_far += current_batch

    print(f"Successfully saved {sample_id} MaskGIT samples to {out_dir}")

if __name__ == '__main__':
    generate_samples()