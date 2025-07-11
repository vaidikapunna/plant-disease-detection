import os
import shutil
import random
source_dir = 'dataset'
train_dir = os.path.join(source_dir, 'train')
val_dir = os.path.join(source_dir, 'val')
split_ratio = 0.8  
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    if not os.path.isdir(class_path) or class_name in ['train', 'val']:
        continue

    print(f"Processing: {class_name}")
    images = os.listdir(class_path)
    random.shuffle(images)
    split_index = int(len(images) * split_ratio)
    train_images = images[:split_index]
    val_images = images[split_index:]
    train_class_dir = os.path.join(train_dir, class_name)
    val_class_dir = os.path.join(val_dir, class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)
    for img in train_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(train_class_dir, img)
        shutil.copyfile(src, dst)
    for img in val_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(val_class_dir, img)
        shutil.copyfile(src, dst)

print(" Done! Your dataset is now split into 'train' and 'val' folders.")