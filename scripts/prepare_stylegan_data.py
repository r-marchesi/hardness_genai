import os
import json
from torchvision import datasets

def prepare_data():
    out_dir = '../../data/cifar100_raw_folders'
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading CIFAR-100...")
    dataset_path = '/public_datasets/PublicDatasets/cifar-100/'
    dataset = datasets.CIFAR100(root=dataset_path, train=True, download=False)
    
    # We will store the label mapping here
    dataset_labels = []
    
    print(f"Exporting {len(dataset)} images and generating dataset.json...")
    for i, (img, label) in enumerate(dataset):
        # Create a folder for the specific class
        class_folder = f'{label:03d}'
        class_dir = os.path.join(out_dir, class_folder)
        os.makedirs(class_dir, exist_ok=True)
        
        # Save the image
        img_name = f'img_{i:05d}.png'
        save_path = os.path.join(class_dir, img_name)
        img.save(save_path)
        
        # Add the relative path and label to our json list
        relative_path = f'{class_folder}/{img_name}'
        dataset_labels.append([relative_path, label])
        
        if (i + 1) % 10000 == 0:
            print(f"Exported {i + 1} / {len(dataset)} images.")
            
    # Save the required dataset.json file at the root of the raw folder
    json_path = os.path.join(out_dir, 'dataset.json')
    with open(json_path, 'w') as f:
        json.dump({"labels": dataset_labels}, f)
        
    print(f"Done! Raw images and dataset.json saved to {out_dir}")

if __name__ == '__main__':
    prepare_data()