import os
import json
import matplotlib
matplotlib.use('Agg')  # Headless backend for cluster environments
import matplotlib.pyplot as plt

# Dynamically resolve PROJECT_ROOT
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

# Base directory where the checkpoints and metrics are saved
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs/synthetic_data/cifar100_checkpoints")

def plot_model_metrics(model_name):
    metrics_file = os.path.join(OUT_DIR, model_name, "metrics", "summary_metrics.json")
    
    if not os.path.exists(metrics_file):
        print(f"Metrics file not found for {model_name}: {metrics_file}")
        return

    with open(metrics_file, 'r') as f:
        data = json.load(f)

    steps = []
    fids = []
    is_means = []
    is_stds = []

    for snapshot_id, metrics in data.items():
        # Parse the step integer from the snapshot_id (e.g., "snapshot_010080" -> 10080)
        try:
            step = int(snapshot_id.split("_")[1])
            steps.append(step)
            fids.append(metrics["Frechet_Inception_Distance"])
            is_means.append(metrics["Inception_Score_Mean"])
            is_stds.append(metrics["Inception_Score_Std"])
        except Exception as e:
            print(f"Error parsing {snapshot_id}: {e}")

    if not steps:
        print(f"No valid data found for {model_name}.")
        return

    # Sort arrays by step to ensure chronologically correct line plotting
    sorted_indices = sorted(range(len(steps)), key=lambda k: steps[k])
    steps = [steps[i] for i in sorted_indices]
    fids = [fids[i] for i in sorted_indices]
    is_means = [is_means[i] for i in sorted_indices]
    is_stds = [is_stds[i] for i in sorted_indices]

    # Create side-by-side plots (FID left, IS right)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 1. FID Plot
    ax1.plot(steps, fids, marker='o', linestyle='-', color='purple', alpha=0.8, linewidth=2)
    ax1.set_title(f"{model_name.upper()} - Fréchet Inception Distance (↓)")
    ax1.set_xlabel("Training Progress (Steps/kimg)")
    ax1.set_ylabel("FID Score (Lower is better)")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # 2. IS Plot
    ax2.errorbar(steps, is_means, yerr=is_stds, marker='s', linestyle='-', color='teal', alpha=0.8, linewidth=2, capsize=5)
    ax2.set_title(f"{model_name.upper()} - Inception Score (↑)")
    ax2.set_xlabel("Training Progress (Steps/kimg)")
    ax2.set_ylabel("IS (Higher is better)")
    ax2.grid(True, linestyle="--", alpha=0.6)

    fig.suptitle(f"{model_name.upper()} Generative Quality Over Training Time", fontsize=16, y=1.02)
    plt.tight_layout()
    
    out_path = os.path.join(OUT_DIR, f"{model_name}_metrics_curve.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {model_name.upper()} metrics plot to: {out_path}")

if __name__ == "__main__":
    print("\n========================================")
    print(" Plotting Generative Metrics (FID & IS)")
    print("========================================")
    
    plot_model_metrics("stylegan")
    plot_model_metrics("edm")
    print("\nDone!")