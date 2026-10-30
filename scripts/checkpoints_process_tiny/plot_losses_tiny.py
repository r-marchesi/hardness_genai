import os
import glob
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs/synthetic_data/tinyimagenet_checkpoints")
os.makedirs(OUT_DIR, exist_ok=True)

STYLEGAN_SNAPSHOTS = [
    5040, 10080, 15120, 20160, 25200, 30240, 35280, 40320, 45360, 50400,
    55440, 60480, 65520
]

EDM_SNAPSHOTS = [
    2509, 5018, 7526, 10035, 12544, 15053, 17562, 20070, 22579, 25088, 27597
]

def get_scalar(val):
    if isinstance(val, dict) and "mean" in val:
        return float(val["mean"])
    return float(val)

def find_closest_kimg(target_kimg, kimg_list):
    return min(kimg_list, key=lambda x: abs(x - target_kimg))

def process_stylegan():
    print("\n========================================")
    print(" Analyzing StyleGAN Tiny-ImageNet Losses")
    print("========================================")
    
    sg_base = os.path.join(PROJECT_ROOT, "checkpoints/stylegan_tiny")
    run_folders = sorted(glob.glob(os.path.join(sg_base, "0000*")))
    
    data = {}
    for folder in run_folders:
        stats_file = os.path.join(folder, "stats.jsonl")
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    if "Progress/kimg" in entry:
                        kimg = get_scalar(entry["Progress/kimg"])
                        g_loss = get_scalar(entry.get("Loss/G/loss", 0))
                        d_loss = get_scalar(entry.get("Loss/D/loss", 0))
                        data[kimg] = {"G_loss": g_loss, "D_loss": d_loss}
    
    if not data:
        print("No StyleGAN stats found.")
        return

    sorted_kimgs = sorted(data.keys())
    g_losses = [data[k]['G_loss'] for k in sorted_kimgs]
    d_losses = [data[k]['D_loss'] for k in sorted_kimgs]

    ckpt_kimgs, ckpt_g_losses, ckpt_d_losses = [], [], []
    for target in STYLEGAN_SNAPSHOTS:
        closest = find_closest_kimg(target, sorted_kimgs)
        ckpt_kimgs.append(closest)
        ckpt_g_losses.append(data[closest]['G_loss'])
        ckpt_d_losses.append(data[closest]['D_loss'])

    plt.figure(figsize=(10, 6))
    plt.plot(sorted_kimgs, g_losses, label="Generator Loss", color="blue", alpha=0.6)
    plt.plot(sorted_kimgs, d_losses, label="Discriminator Loss", color="orange", alpha=0.6)
    plt.scatter(ckpt_kimgs, ckpt_g_losses, color="blue", edgecolor='black', zorder=5, label="G Checkpoints")
    plt.scatter(ckpt_kimgs, ckpt_d_losses, color="orange", edgecolor='black', zorder=5, label="D Checkpoints")
    
    plt.title("StyleGAN Training Losses (Tiny-ImageNet)")
    plt.xlabel("Training Progress (kimg)")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    
    out_path = os.path.join(OUT_DIR, "stylegan_loss_curve.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved StyleGAN loss plot to: {out_path}")

def process_edm():
    print("\n========================================")
    print(" Analyzing EDM Tiny-ImageNet Losses")
    print("========================================")
    
    edm_base = os.path.join(PROJECT_ROOT, "checkpoints/edm_tiny")
    run_folders = sorted(glob.glob(os.path.join(edm_base, "0000*")))
    
    data = {}
    for folder in run_folders:
        stats_file = os.path.join(folder, "stats.jsonl")
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    if "Progress/kimg" in entry and "Loss/loss" in entry:
                        kimg = get_scalar(entry["Progress/kimg"])
                        loss = get_scalar(entry["Loss/loss"])
                        data[kimg] = loss

    if not data:
        print("No EDM stats found.")
        return

    sorted_kimgs = sorted(data.keys())
    losses = [data[k] for k in sorted_kimgs]

    ckpt_kimgs, ckpt_losses = [], []
    for target in EDM_SNAPSHOTS:
        closest = find_closest_kimg(target, sorted_kimgs)
        ckpt_kimgs.append(closest)
        ckpt_losses.append(data[closest])

    plt.figure(figsize=(10, 6))
    plt.plot(sorted_kimgs, losses, label="EDM Training Loss", color="green", alpha=0.7)
    plt.scatter(ckpt_kimgs, ckpt_losses, color="red", edgecolor='black', zorder=5, label="Checkpoints")
    
    plt.title("EDM Training Loss (Tiny-ImageNet)")
    plt.xlabel("Training Progress (kimg)")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    
    out_path = os.path.join(OUT_DIR, "edm_loss_curve.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved EDM loss plot to: {out_path}")

if __name__ == "__main__":
    process_stylegan()
    process_edm()