import os
import sys
import json
import torch
import argparse
import torch_fidelity
from PIL import Image
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

STYLEGAN_REPO = os.path.join(PROJECT_ROOT, "repos_genai/stylegan3")
sys.path.insert(0, STYLEGAN_REPO)

import dnnlib
import legacy

# 5,040 kimg progression up to current max (066528)
DEFAULT_SNAPSHOTS = [
    "network-snapshot-005040.pkl", "network-snapshot-010080.pkl",
    "network-snapshot-015120.pkl", "network-snapshot-020160.pkl",
    "network-snapshot-025200.pkl", "network-snapshot-030240.pkl",
    "network-snapshot-035280.pkl", "network-snapshot-040320.pkl",
    "network-snapshot-045360.pkl", "network-snapshot-050400.pkl",
    "network-snapshot-055440.pkl", "network-snapshot-060480.pkl",
    "network-snapshot-065520.pkl"
]

def find_snapshot_path(base_dir, snapshot_name):
    """Recursively search for a snapshot file in the base directory."""
    for root, dirs, files in os.walk(base_dir):
        if snapshot_name in files:
            return os.path.join(root, snapshot_name)
    return None

def generate_balanced_dataset(G, out_dir, batch_size=100, samples_per_class=250, num_classes=200, device='cuda'):
    os.makedirs(out_dir, exist_ok=True)
    print(f"  -> Generating BALANCED dataset ({samples_per_class} per class across {num_classes} classes)...")
    
    for label in tqdm(range(num_classes), desc="    Balanced Dataset", leave=False):
        class_dir = os.path.join(out_dir, str(label))
        os.makedirs(class_dir, exist_ok=True)
        
        generated_so_far = 0
        while generated_so_far < samples_per_class:
            current_batch = min(batch_size, samples_per_class - generated_so_far)
            
            z = torch.randn([current_batch, G.z_dim], device=device)
            c = torch.zeros([current_batch, G.c_dim], device=device)
            c[:, label] = 1.0
            
            with torch.no_grad():
                img = G(z, c)
                
            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img = img.permute(0, 2, 3, 1).cpu().numpy()
            
            for i in range(current_batch):
                pil_img = Image.fromarray(img[i], 'RGB')
                pil_img.save(os.path.join(class_dir, f"img_{generated_so_far + i:04d}.jpg"))
                
            generated_so_far += current_batch

def process_checkpoints(ckpt_base_dir, real_data_dir, output_base_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running StyleGAN Tiny-ImageNet checkpoint sweep on: {device}")
    
    balanced_base_dir = os.path.join(output_base_dir, "balanced")
    metrics_base_dir = os.path.join(output_base_dir, "metrics")
    
    os.makedirs(metrics_base_dir, exist_ok=True)
    
    # Incremental state loading
    summary_path = os.path.join(metrics_base_dir, "summary_metrics.json")
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            all_metrics = json.load(f)
        print(f"Loaded {len(all_metrics)} existing metrics. Incremental mode active.")
    else:
        all_metrics = {}

    for snapshot in DEFAULT_SNAPSHOTS:
        snapshot_id = snapshot.replace(".pkl", "").replace("network-snapshot-", "snapshot_")
        
        # Skip if already fully processed
        if snapshot_id in all_metrics and os.path.exists(os.path.join(balanced_base_dir, snapshot_id)):
            print(f"Skipping {snapshot}: Already processed and merged.")
            continue

        ckpt_path = find_snapshot_path(ckpt_base_dir, snapshot)
        
        if not ckpt_path:
            print(f"Skipping {snapshot}: Checkpoint file not found anywhere inside {ckpt_base_dir}")
            continue

        print(f"\n========================================================")
        print(f" Processing StyleGAN Checkpoint: {snapshot}")
        print(f" Found at: {ckpt_path}")
        print(f"========================================================")

        snapshot_balanced_dir = os.path.join(balanced_base_dir, snapshot_id)
        
        with open(ckpt_path, 'rb') as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device)

        generate_balanced_dataset(G, snapshot_balanced_dir, device=device)

        print(f"  -> Computing IS & FID metrics on balanced dataset...")
        metrics = torch_fidelity.calculate_metrics(
            input1=snapshot_balanced_dir,
            input2=real_data_dir,
            cuda=True,
            isc=True,
            fid=True,
            samples_find_deep=True,
            verbose=False
        )

        ckpt_metrics = {
            "Inception_Score_Mean": metrics.get("inception_score_mean"),
            "Inception_Score_Std": metrics.get("inception_score_std"),
            "Frechet_Inception_Distance": metrics.get("frechet_inception_distance")
        }

        print(f"  Results for {snapshot_id}:")
        print(f"    - IS:  {ckpt_metrics['Inception_Score_Mean']:.4f} ± {ckpt_metrics['Inception_Score_Std']:.4f}")
        print(f"    - FID: {ckpt_metrics['Frechet_Inception_Distance']:.4f}")

        with open(os.path.join(metrics_base_dir, f"{snapshot_id}.json"), 'w') as f:
            json.dump(ckpt_metrics, f, indent=4)

        # Update and save incrementally
        all_metrics[snapshot_id] = ckpt_metrics
        with open(summary_path, 'w') as f:
            json.dump(all_metrics, f, indent=4)

    print(f"\nCompleted StyleGAN Tiny-ImageNet checkpoint sweep! Summary saved to {summary_path}")

if __name__ == "__main__":
    default_ckpt_base_dir = os.path.join(PROJECT_ROOT, "checkpoints/stylegan_tiny")
    default_real_dir = os.path.join(PROJECT_ROOT, "data/tinyimagenet_formatted")
    default_out_base = os.path.join(PROJECT_ROOT, "outputs/synthetic_data/tinyimagenet_checkpoints/stylegan")

    parser = argparse.ArgumentParser(description="Generate and evaluate StyleGAN Tiny-ImageNet checkpoints")
    parser.add_argument("--ckpt_base_dir", type=str, default=default_ckpt_base_dir)
    parser.add_argument("--real_dir", type=str, default=default_real_dir)
    parser.add_argument("--out_base", type=str, default=default_out_base)

    args = parser.parse_args()
    process_checkpoints(args.ckpt_base_dir, args.real_dir, args.out_base)