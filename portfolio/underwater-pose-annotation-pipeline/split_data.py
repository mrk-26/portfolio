import json
import os
import random
import shutil
from tqdm import tqdm

# ==== CONFIGURATION ====
json_path = 'custom_underwater_dataset/annotations/person_keypoints_default.json'  # Path to your original JSON file
image_dir = 'custom_underwater_dataset/images'                    # Where all original images are stored
output_dir = 'custom_underwater_dataset'                             # Root output folder (current dir)
split_ratio = [0.7, 0.15, 0.15]              # Train, val, test split
# =======================

# Load JSON
with open(json_path) as f:
    coco = json.load(f)

images = coco['images']
annotations = coco['annotations']

# Shuffle images
random.shuffle(images)

# Split indices
n = len(images)
n_train = int(n * split_ratio[0])
n_val = int(n * split_ratio[1])
train_imgs = images[:n_train]
val_imgs = images[n_train:n_train+n_val]
test_imgs = images[n_train+n_val:]

# Helper: Get annotations for image_ids
def filter_annotations(image_list):
    img_ids = {img['id'] for img in image_list}
    return [ann for ann in annotations if ann['image_id'] in img_ids]

# Split annotations
train = {
    'images': train_imgs,
    'annotations': filter_annotations(train_imgs),
    'categories': coco['categories']
}
val = {
    'images': val_imgs,
    'annotations': filter_annotations(val_imgs),
    'categories': coco['categories']
}
test = {
    'images': test_imgs,
    'annotations': filter_annotations(test_imgs),
    'categories': coco['categories']
}

# Output directories
os.makedirs(f'{output_dir}/images/train', exist_ok=True)
os.makedirs(f'{output_dir}/images/val', exist_ok=True)
os.makedirs(f'{output_dir}/images/test', exist_ok=True)
os.makedirs(f'{output_dir}/annotations', exist_ok=True)

# Copy images
def copy_images(split_imgs, split_name):
    for img in tqdm(split_imgs, desc=f'Copying {split_name} images'):
        src_path = os.path.join(image_dir, img['file_name'])
        dst_path = os.path.join(output_dir, f'images/{split_name}', img['file_name'])
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
        else:
            print(f"Warning: {src_path} does not exist!")

copy_images(train_imgs, 'train')
copy_images(val_imgs, 'val')
copy_images(test_imgs, 'test')

# Save JSON files
with open(f'{output_dir}/annotations/train.json', 'w') as f:
    json.dump(train, f)
with open(f'{output_dir}/annotations/val.json', 'w') as f:
    json.dump(val, f)
with open(f'{output_dir}/annotations/test.json', 'w') as f:
    json.dump(test, f)

print("✅ Done: Images copied and JSONs saved.")
