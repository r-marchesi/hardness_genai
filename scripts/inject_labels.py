import os
import json
import zipfile

PROJECT_ROOT = "/storage/DSH/projects/hardness_genai"
FORMATTED_DIR = os.path.join(PROJECT_ROOT, "data/tinyimagenet_formatted")
ZIP_PATH = os.path.join(PROJECT_ROOT, "data/tinyimagenet_custom.zip")
JSON_PATH = os.path.join(PROJECT_ROOT, "data/dataset.json")

print("Scanning folders to generate label mappings...")
labels = []
for class_folder in sorted(os.listdir(FORMATTED_DIR)):
    if class_folder.isdigit():
        label_idx = int(class_folder)
        class_path = os.path.join(FORMATTED_DIR, class_folder)
        
        for img_name in sorted(os.listdir(class_path)):
            if img_name.endswith('.jpg'):
                # Format required by NVIDIA: ["000/img_000000.jpg", 0]
                rel_path = f"{class_folder}/{img_name}"
                labels.append([rel_path, label_idx])

# Write the JSON file to disk
dataset_json = {"labels": labels}
with open(JSON_PATH, 'w') as f:
    json.dump(dataset_json, f)

# Append the JSON file into the existing dataset zip
print(f"Injecting dataset.json into {ZIP_PATH}...")
with zipfile.ZipFile(ZIP_PATH, 'a') as zipf:
    zipf.write(JSON_PATH, arcname="dataset.json")

print("Success! dataset.json injected. Models are ready for conditional training.")