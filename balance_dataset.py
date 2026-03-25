import os
import random
import shutil
from PIL import Image, ImageOps

def augment_image(image):
    if random.random() > 0.5:
        image = ImageOps.mirror(image)
    if random.random() > 0.5:
        image = ImageOps.flip(image)
    angle = random.choice([0, 90, 180, 270])
    if angle != 0:
        image = image.rotate(angle)
    if random.random() > 0.5:
        image = image.rotate(random.uniform(-15, 15))
    return image

def get_class_images(src_dir, cls, merge_map):
    # Returns a list of full paths for a logical class
    images = []
    for src_cls, dest_cls in merge_map.items():
        if dest_cls == cls:
            cls_dir = os.path.join(src_dir, src_cls)
            if os.path.exists(cls_dir):
                cls_imgs = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                images.extend(cls_imgs)
    return images

def balance_classes(src_dir, dest_dir, merge_map, target_count=500):
    os.makedirs(dest_dir, exist_ok=True)
    unique_dest_classes = set(merge_map.values())
    
    for dest_cls in unique_dest_classes:
        target_cls_dir = os.path.join(dest_dir, dest_cls)
        os.makedirs(target_cls_dir, exist_ok=True)
        images = get_class_images(src_dir, dest_cls, merge_map)
        
        if not images:
            print(f"No images found for {dest_cls}")
            continue

        if len(images) >= target_count:
            sampled = random.sample(images, target_count)
            for i, img_path in enumerate(sampled):
                ext = os.path.splitext(img_path)[1]
                dest_path = os.path.join(target_cls_dir, f"{dest_cls}_{i}{ext}")
                shutil.copy2(img_path, dest_path)
            print(f"[{dest_cls}] Downsampled from {len(images)} to {target_count}")
        else:
            for i, img_path in enumerate(images):
                ext = os.path.splitext(img_path)[1]
                dest_path = os.path.join(target_cls_dir, f"orig_{i}{ext}")
                shutil.copy2(img_path, dest_path)
            
            current_count = len(images)
            augment_count = target_count - current_count
            print(f"[{dest_cls}] Upsampling from {current_count} to {target_count} (Generating {augment_count})")
            
            for i in range(augment_count):
                base_img_path = random.choice(images)
                ext = os.path.splitext(base_img_path)[1]
                try:
                    with Image.open(base_img_path) as img:
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        aug_img = augment_image(img)
                        aug_name = f"aug_{i}{ext}"
                        aug_img.save(os.path.join(target_cls_dir, aug_name))
                except Exception as e:
                    print(f"Error augmenting {base_img_path}: {e}")

def main():
    source_base = r"C:\Projects\HCLTech\Cosmetic Defect Detection Project\Metal Surface defect datas\industrial_defect_dataset"
    target_base = r"C:\Projects\HCLTech\Cosmetic Defect Detection Project\Metal Surface defect datas\industrial_defect_dataset_balanced"
    
    src_train = os.path.join(source_base, "train")
    target_train = os.path.join(target_base, "train")
    src_val = os.path.join(source_base, "val")
    target_val = os.path.join(target_base, "val")
    
    if os.path.exists(target_base):
        print("Cleaning up old balanced directory...")
        shutil.rmtree(target_base)
        
    print("--- Balancing Training Data ---")
    classes = [d for d in os.listdir(src_train) if os.path.isdir(os.path.join(src_train, d))]
    
    merge_map = {}
    for c in classes:
        if c == "punching_hole":
            merge_map[c] = "hole"
        else:
            merge_map[c] = c
            
    balance_classes(src_train, target_train, merge_map, target_count=500)
    
    print("\n--- Copying Validation Data (Merged) ---")
    os.makedirs(target_val, exist_ok=True)
    val_classes = [d for d in os.listdir(src_val) if os.path.isdir(os.path.join(src_val, d))]
    
    val_merge_map = {}
    for c in val_classes:
        if c == "punching_hole":
            val_merge_map[c] = "hole"
        else:
            val_merge_map[c] = c
            
    unique_val_dest_classes = set(val_merge_map.values())
    for dest_cls in unique_val_dest_classes:
        target_cls_dir = os.path.join(target_val, dest_cls)
        os.makedirs(target_cls_dir, exist_ok=True)
        images = get_class_images(src_val, dest_cls, val_merge_map)
        for i, img_path in enumerate(images):
            ext = os.path.splitext(img_path)[1]
            dest_path = os.path.join(target_cls_dir, f"{os.path.basename(os.path.dirname(img_path))}_{i}{ext}")
            shutil.copy2(img_path, dest_path)
            
    print("\n✅ Dataset balancing complete!")
    print(f"New dataset is located at: {target_base}")

if __name__ == '__main__':
    main()
