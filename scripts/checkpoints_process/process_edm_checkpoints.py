import os
import sys
import json
import torch
import pickle
import argparse
import torch_fidelity
from PIL import Image
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

sys.path.insert(0, os.path.join(PROJECT_ROOT, "repos_genai/edm"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
sys.path.insert(0, PROJECT_ROOT)

from generate import edm_sampler

# 28 distributed snapshots across all EDM training runs
DEFAULT_SNAPSHOTS = [
    # Old targets
    "network-snapshot-010035.pkl", "network-snapshot-020070.pkl",
    "network-snapshot-030106.pkl", "network-snapshot-040141.pkl",
    "network-snapshot-050176.pkl", "network-snapshot-060211.pkl",
    "network-snapshot-070246.pkl", "network-snapshot-080282.pkl",
    "network-snapshot-090317.pkl", "network-snapshot-100352.pkl",
    "network-snapshot-110387.pkl", "network-snapshot-120422.pkl",
    "network-snapshot-130458.pkl", "network-snapshot-140493.pkl",
    "network-snapshot-150528.pkl", "network-snapshot-160563.pkl",
    "network-snapshot-170598.pkl", "network-snapshot-180634.pkl",
    "network-snapshot-190669.pkl", "network-snapshot-200000.pkl",
    # New targets
    "network-snapshot-210035.pkl", "network-snapshot-220070.pkl", 
    "network-snapshot-230106.pkl", "network-snapshot-240141.pkl", 
    "network-snapshot-250176.pkl", "network-snapshot-260211.pkl", 
    "network-snapshot-270246.pkl", "network-snapshot-280282.pkl"
]

def find_snapshot_path(base_dir, snapshot_name):
    """Recursively search for a snapshot file in the base directory."""
    for root, dirs, files in os.walk(base_dir):
        if snapshot_name in files:
            return os.path.join(root, snapshot_name)
    return None

def generate_balanced_dataset(net, out_dir, batch_size=100, samples_per_class=500, num_classes=100, device='cuda'):
    os.makedirs(out_dir, exist_ok=True)
    print(f"  -> Generating BALANCED dataset ({samples_per_class} per class)...")
    
    for label in tqdm(range(num_classes), desc="    Balanced Dataset", leave=False):
        class_dir = os.path.join(out_dir, str(label))
        os.makedirs(class_dir, exist_ok=True)
        
        generated_so_far = 0
        while generated_so_far < samples_per_class:
            current_batch = min(batch_size, samples_per_class - generated_so_far)
            
            latents = torch.randn([current_batch, net.img_channels, net.img_resolution, net.img_resolution], device=device)
            class_labels = torch.zeros([current_batch, net.label_dim], device=device)
            class_labels[:, label] = 1.0
            
            with torch.no_grad():
                images = edm_sampler(net, latents, class_labels, num_steps=18)
                
            images = (images * 127.5 + 128).clip(0, 255).to(torch.uint8)
            images = images.permute(0, 2, 3, 1).cpu().numpy() 
            
            for i in range(current_batch):
                pil_img = Image.fromarray(images[i], 'RGB')
                pil_img.save(os.path.join(class_dir, f"img_{generated_so_far + i:04d}.jpg"))
                
            generated_so_far += current_batch

def generate_oversampling_dataset(net, json_path, out_dir, batch_size=100, device='cuda'):
    os.makedirs(out_dir, exist_ok=True)
    print(f"  -> Generating OVERSAMPLING dataset (exact targets from JSON)...")
    
    with open(json_path, 'r') as f:
        resampling_ratios = json.load(f)
        
    for label, count in enumerate(tqdm(resampling_ratios, desc="    Oversampling Dataset", leave=False)):
        target_count = int(count)  
        if target_count <= 0:
            continue
            
        class_dir = os.path.join(out_dir, str(label))
        os.makedirs(class_dir, exist_ok=True)
        
        generated_so_far = 0
        while generated_so_far < target_count:
            current_batch = min(batch_size, target_count - generated_so_far)
            
            latents = torch.randn([current_batch, net.img_channels, net.img_resolution, net.img_resolution], device=device)
            class_labels = torch.zeros([current_batch, net.label_dim], device=device)
            class_labels[:, label] = 1.0
            
            with torch.no_grad():
                images = edm_sampler(net, latents, class_labels, num_steps=18)
                
            images = (images * 127.5 + 128).clip(0, 255).to(torch.uint8)
            images = images.permute(0, 2, 3, 1).cpu().numpy() 
            
            for i in range(current_batch):
                pil_img = Image.fromarray(images[i], 'RGB')
                pil_img.save(os.path.join(class_dir, f"img_{generated_so_far + i:04d}.jpg"))
                
            generated_so_far += current_batch

def process_checkpoints(ckpt_base_dir, real_data_dir, json_path, output_base_dir, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running checkpoint sweep on: {device}")
    
    balanced_base_dir = os.path.join(output_base_dir, "balanced")
    oversampling_base_dir = os.path.join(output_base_dir, "oversampling")
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
        print(f" Processing EDM Checkpoint: {snapshot}")
        print(f" Found at: {ckpt_path}")
        print(f"========================================================")

        snapshot_balanced_dir = os.path.join(balanced_base_dir, snapshot_id)
        snapshot_oversampling_dir = os.path.join(oversampling_base_dir, snapshot_id)
        
        with open(ckpt_path, 'rb') as f:
            net = pickle.load(f)['ema'].to(device)

        generate_balanced_dataset(net, snapshot_balanced_dir, batch_size=batch_size, device=device)
        generate_oversampling_dataset(net, json_path, snapshot_oversampling_dir, batch_size=batch_size, device=device)

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

    print(f"\nCompleted EDM checkpoint sweep! Summary saved to {summary_path}")

if __name__ == "__main__":
    default_ckpt_base_dir = os.path.join(PROJECT_ROOT, "checkpoints/edm")
    
    default_real_dir = os.path.join(PROJECT_ROOT, "outputs/training_data")
    default_json = os.path.join(PROJECT_ROOT, "data/oversampling_targets.json")
    default_out_base = "/storage/DSH/projects/hardness_genai/outputs/synthetic_data/cifar100_checkpoints/edm"

    parser = argparse.ArgumentParser(description="Generate and evaluate EDM checkpoints")
    parser.add_argument("--ckpt_base_dir", type=str, default=default_ckpt_base_dir, help="Base directory to recursively search for snapshots")
    parser.add_argument("--real_dir", type=str, default=default_real_dir)
    parser.add_argument("--json_path", type=str, default=default_json)
    parser.add_argument("--out_base", type=str, default=default_out_base)
    parser.add_argument("--batch_size", type=int, default=100)

    args = parser.parse_args()
    process_checkpoints(args.ckpt_base_dir, args.real_dir, args.json_path, args.out_base, args.batch_size)