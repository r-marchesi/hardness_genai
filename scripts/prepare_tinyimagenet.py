import os
import zipfile
from tqdm import tqdm

# Auto-install the datasets library if it's missing in the container
try:
    from datasets import load_dataset
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
    from datasets import load_dataset

PROJECT_ROOT = "/storage/DSH/projects/hardness_genai"
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FORMATTED_DIR = os.path.join(DATA_DIR, "tinyimagenet_formatted")
os.makedirs(FORMATTED_DIR, exist_ok=True)

print("Downloading Tiny-ImageNet from Hugging Face API...")
# This safely pulls the official 64x64 images without relying on Stanford's servers
dataset = load_dataset("zh-plus/tiny-imagenet", split="train")

print("Saving images directly to class folders...")
for i, item in enumerate(tqdm(dataset)):
    img = item['image']
    label = item['label']
    
    # StyleGAN expects strict numerical folders: 000, 001, etc.
    class_dir = os.path.join(FORMATTED_DIR, f"{label:03d}")
    os.makedirs(class_dir, exist_ok=True)
    
    img_path = os.path.join(class_dir, f"img_{i:06d}.jpg")
    
    # Ensure standard RGB format
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    img.save(img_path)

FINAL_ZIP = os.path.join(DATA_DIR, "tinyimagenet_custom.zip")
print(f"Creating final dataset zip at {FINAL_ZIP}...")
with zipfile.ZipFile(FINAL_ZIP, "w", zipfile.ZIP_STORED) as zipf:
    for root, _, files in os.walk(FORMATTED_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, FORMATTED_DIR)
            zipf.write(file_path, arcname)

print("Tiny-ImageNet preparation complete!")