import os
import json
import zipfile
import io
from torchvision import datasets

def build_dataset():
    print("Loading CIFAR-100 via torchvision...")
    # Load the dataset using torchvision (which automatically handles the pickled tar.gz)
    dataset = datasets.CIFAR100(root='/public_datasets/PublicDatasets/cifar-100/', train=True, download=False)

    zip_path = 'data/cifar100_custom.zip'
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)

    labels = []
    print(f"Packaging {len(dataset)} images into {zip_path}...")

    # Write directly to a ZIP file to save disk I/O operations
    with zipfile.ZipFile(zip_path, 'w') as zf:
        for idx, (img, label) in enumerate(dataset):
            # StyleGAN standard: group images into folders of 1000
            img_name = f'{idx // 1000:05d}/img{idx:08d}.png'
            
            # Save PIL image to memory buffer, then write to ZIP
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            zf.writestr(img_name, buf.getvalue())
            
            # Record the label for dataset.json
            labels.append([img_name, label])
            
        # Write the JSON label file required for class-conditioning (--cond=1)
        zf.writestr('dataset.json', json.dumps({'labels': labels}))

    print(f"Done! Created dataset with {len(labels)} conditional labels.")

if __name__ == "__main__":
    build_dataset()