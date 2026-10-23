import os
import glob
import json
import matplotlib
matplotlib.use('Agg')  # Headless backend for cluster environments
import matplotlib.pyplot as plt

# Dynamically resolve PROJECT_ROOT
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs/synthetic_data/cifar100_checkpoints")
os.makedirs(OUT_DIR, exist_ok=True)

# Define the exact snapshots you selected
STYLEGAN_SNAPSHOTS = [
    5040, 10080, 15120, 20160, 25200, 30240, 35280, 40320, 45360, 50400,
    55440, 60480, 65520, 70560, 75600, 80640, 85680, 90720, 95760, 100800,
    105840, 110880, 115920, 120960, 126000, 131040, 136080, 141120, 146160, 151200, 156240, 161280, 166320, 171360, 176400, 181440, 186480, 191520, 196560, 201600, 206640, 211680, 216720, 221760
]

EDM_SNAPSHOTS = [
    10035, 20070, 30106, 40141, 50176, 60211, 70246, 80282, 90317, 100352,
    110387, 120422, 130458, 140493, 150528, 160563, 170598, 180634, 190669, 200000,
    210035, 220070, 230106, 240141, 250176, 260211, 270246, 280282
]

def get_scalar(val):
    """Extracts the float value if NVIDIA's logger saved it as a dict with 'mean'."""
    if isinstance(val, dict) and "mean" in val:
        return float(val["mean"])
    return float(val)

def find_closest_kimg(target_kimg, kimg_list):
    """Finds the closest recorded kimg step to the snapshot target."""
    return min(kimg_list, key=lambda x: abs(x - target_kimg))

def process_stylegan():
    print("\n========================================")
    print(" Analyzing StyleGAN Training Losses")
    print("========================================")
    
    sg_dir = os.path.join(PROJECT_ROOT, "checkpoints/stylegan/00027-stylegan2-cifar100_stylegan-gpus1-batch64-gamma0.01")
    stats_file = os.path.join(sg_dir, "stats.jsonl")
    
    data = {}
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

    # Find stats for chosen checkpoints
    print(f"{'Snapshot (kimg)':<18} | {'Generator Loss':<15} | {'Discriminator Loss'}")
    print("-" * 55)
    
    ckpt_kimgs = []
    ckpt_g_losses = []
    ckpt_d_losses = []
    
    for target in STYLEGAN_SNAPSHOTS:
        closest = find_closest_kimg(target, sorted_kimgs)
        ckpt_kimgs.append(closest)
        ckpt_g_losses.append(data[closest]['G_loss'])
        ckpt_d_losses.append(data[closest]['D_loss'])
        print(f"{target:<18} | {data[closest]['G_loss']:<15.4f} | {data[closest]['D_loss']:.4f}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_kimgs, g_losses, label="Generator Loss", color="blue", alpha=0.6)
    plt.plot(sorted_kimgs, d_losses, label="Discriminator Loss", color="orange", alpha=0.6)
    plt.scatter(ckpt_kimgs, ckpt_g_losses, color="blue", edgecolor='black', zorder=5, label="G Checkpoints")
    plt.scatter(ckpt_kimgs, ckpt_d_losses, color="orange", edgecolor='black', zorder=5, label="D Checkpoints")
    
    plt.title("StyleGAN Training Losses")
    plt.xlabel("Training Progress (kimg)")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    
    out_path = os.path.join(OUT_DIR, "stylegan_loss_curve.png")
    plt.savefig(out_path, dpi=300)
    print(f"\nSaved StyleGAN loss plot to: {out_path}")

def process_edm():
    print("\n========================================")
    print(" Analyzing EDM Training Losses")
    print("========================================")
    
    edm_base = os.path.join(PROJECT_ROOT, "checkpoints/edm")
    # Grab all 00000, 00001, 00002 folders and sort them sequentially
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
                        # By writing to a dict sequentially, later restarts seamlessly 
                        # overwrite any overlapping kimg steps from a crashed run.
                        data[kimg] = loss

    if not data:
        print("No EDM stats found.")
        return

    sorted_kimgs = sorted(data.keys())
    losses = [data[k] for k in sorted_kimgs]

    # Find stats for chosen checkpoints
    print(f"{'Snapshot (kimg)':<18} | {'EDM Loss'}")
    print("-" * 35)
    
    ckpt_kimgs = []
    ckpt_losses = []
    
    for target in EDM_SNAPSHOTS:
        closest = find_closest_kimg(target, sorted_kimgs)
        ckpt_kimgs.append(closest)
        ckpt_losses.append(data[closest])
        print(f"{target:<18} | {data[closest]:.4f}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_kimgs, losses, label="EDM Training Loss", color="green", alpha=0.7)
    plt.scatter(ckpt_kimgs, ckpt_losses, color="red", edgecolor='black', zorder=5, label="Checkpoints")
    
    plt.title("EDM Training Loss Across Restarts")
    plt.xlabel("Training Progress (kimg)")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    
    out_path = os.path.join(OUT_DIR, "edm_loss_curve.png")
    plt.savefig(out_path, dpi=300)
    print(f"\nSaved EDM loss plot to: {out_path}")

if __name__ == "__main__":
    process_stylegan()
    process_edm()