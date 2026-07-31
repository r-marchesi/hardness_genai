import os
import json
import argparse
import torch_fidelity

# Dynamically resolve PROJECT_ROOT based on script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../"))

def evaluate_models(real_data_dir, synthetic_base_dir, output_json):
    if not os.path.exists(real_data_dir):
        raise FileNotFoundError(f"Real data directory not found: {real_data_dir}. Please run extract_cifar100_train.py first.")

    # The models you have generated balanced data for
    models_to_evaluate = ["stylegan", "edm", "vae"]
    results = {}

    print(f"--- Starting Evaluation ---")
    print(f"Reference (Real) Data: {real_data_dir}")

    for model in models_to_evaluate:
        synth_dir = os.path.join(synthetic_base_dir, model)
        
        if not os.path.exists(synth_dir):
            print(f"Skipping {model}: Synthetic data folder not found at {synth_dir}")
            continue

        print(f"\n===========================================")
        print(f"Evaluating {model.upper()}...")
        print(f"Synthetic Data: {synth_dir}")
        print(f"===========================================")

        # Compute IS and FID using torch-fidelity
        # torch-fidelity automatically recursively reads images from the given directory
        metrics = torch_fidelity.calculate_metrics(
            input1=synth_dir, 
            input2=real_data_dir, 
            cuda=True, 
            isc=True, 
            fid=True, 
            samples_find_deep=True,  # <-- THE FIX
            verbose=False
        )

        results[model] = {
            "Inception_Score_Mean": metrics.get("inception_score_mean"),
            "Inception_Score_Std": metrics.get("inception_score_std"),
            "Frechet_Inception_Distance": metrics.get("frechet_inception_distance")
        }

        print(f"{model.upper()} Results:")
        print(f"  - IS:  {results[model]['Inception_Score_Mean']:.4f} ± {results[model]['Inception_Score_Std']:.4f}")
        print(f"  - FID: {results[model]['Frechet_Inception_Distance']:.4f}")

    # Save results for your colleague
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nAll evaluations complete. Results saved to {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute IS and FID for generated CIFAR-100 datasets")
    
    # Pointing to the new training_data folder we just created
    default_real_dir = os.path.join(PROJECT_ROOT, "outputs/training_data")
    default_synth_base = os.path.join(PROJECT_ROOT, "outputs/synthetic_data/cifar100_balanced")
    default_output = os.path.join(PROJECT_ROOT, "outputs/evaluation_results_balanced.json")
    
    parser.add_argument("--real_dir", type=str, default=default_real_dir, help="Path to real CIFAR-100 images")
    parser.add_argument("--synth_base_dir", type=str, default=default_synth_base, help="Base path containing model folders")
    parser.add_argument("--out_json", type=str, default=default_output, help="Where to save the JSON results")
    
    args = parser.parse_args()
    evaluate_models(args.real_dir, args.synth_base_dir, args.out_json)