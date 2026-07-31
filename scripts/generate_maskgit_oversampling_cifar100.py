import os
import sys
import json
import torch
from torchvision.utils import save_image
from transformers import PreTrainedTokenizerBase
from muse_maskgit_pytorch import VQGanVAE, MaskGit, MaskGitTransformer

PreTrainedTokenizerBase.batch_encode_plus = PreTrainedTokenizerBase.__call__

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../"))

def generate_samples():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae = VQGanVAE(dim=128, channels=3, layers=2, discr_layers=1, codebook_size=256)
    transformer = MaskGitTransformer(num_tokens=256, seq_len=64, dim=512, depth=6, dim_head=64, heads=8, ff_mult=4, flash=False)
    maskgit = MaskGit(vae=vae, transformer=transformer, image_size=32, cond_drop_prob=0.1).to(device)

    weights_path = os.path.join(PROJECT_ROOT, "checkpoints/maskgit/transformer/maskgit_epoch_250.pt")
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
            
        generated_so_far = 0
        while generated_so_far < target_count:
            current_batch = min(batch_size, target_count - generated_so_far)
            texts = [f"class {label}" for _ in range(current_batch)]
            
            with torch.no_grad():
                generated_imgs = maskgit.generate(texts=texts, cond_scale=1.5)
                
            generated_imgs = generated_imgs.clamp(0, 1)
            
            for i in range(current_batch):
                save_path = os.path.join(out_dir, f"{sample_id}_{label}.jpg")
                save_image(generated_imgs[i], save_path)
                sample_id += 1
                
            generated_so_far += current_batch

if __name__ == '__main__':
    generate_samples()