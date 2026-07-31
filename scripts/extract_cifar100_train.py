import os
import torchvision
from tqdm import tqdm

# Dynamically resolve PROJECT_ROOT based on script location 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../"))

def extract_cifar100_train(out_dir):
    print(f"Loading CIFAR-100 training set...")
    # Load the CIFAR-100 training dataset (downloads to data/ if not present)
    data_dir = os.path.join(PROJECT_ROOT, "data")
    dataset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True)
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Keep track of how many images we've saved per class to name them sequentially
    class_counts = {i: 0 for i in range(100)}
    
    print(f"Extracting 50,000 training images to {out_dir}...")
    
    for i in tqdm(range(len(dataset)), desc="Extracting images"):
        img, label = dataset[i]
        
        # Create class subfolder
        class_dir = os.path.join(out_dir, str(label))
        os.makedirs(class_dir, exist_ok=True)
        
        # Save image matching the synthetic data naming convention
        img_name = f"img_{class_counts[label]:04d}.jpg"
        img.save(os.path.join(class_dir, img_name))
        
        class_counts[label] += 1
        
    print(f"Successfully extracted {len(dataset)} real images across 100 classes.")

if __name__ == "__main__":
    output_directory = os.path.join(PROJECT_ROOT, "outputs/training_data")
    extract_cifar100_train(output_directory)