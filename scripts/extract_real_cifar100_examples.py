import os
from torchvision import datasets

def extract_real_samples():
    out_dir = '../../outputs/real_samples'
    os.makedirs(out_dir, exist_ok=True)
    
    # Load the training dataset directly. 
    # Without transforms, this yields raw PIL Images.
    dataset_path = '/public_datasets/PublicDatasets/cifar-100/'
    dataset = datasets.CIFAR100(root=dataset_path, train=True, download=False)
    
    # We will use a set to track which classes we have already saved
    saved_classes = set()
    
    print('Extracting 100 ground-truth images (one per CIFAR-100 class)...')
    
    # Iterate through the dataset until we find one example of every class
    for img, label in dataset:
        if label not in saved_classes:
            save_path = os.path.join(out_dir, f'real_cifar100_class{label:03d}.png')
            img.save(save_path)
            saved_classes.add(label)
            
        # Stop searching once we have all 100 classes
        if len(saved_classes) == 100:
            break
            
    print(f'Successfully saved 100 real samples to {out_dir}')

if __name__ == '__main__':
    extract_real_samples()