import os
import random
import shutil

def split_dataset_with_test(
    src_images_dir,
    src_labels_dir,
    dest_dir,
    train_ratio=0.8,
    val_ratio=0.2,
    max_samples=None,
    seed=42
):
    random.seed(seed)

    images = sorted([
        f for f in os.listdir(src_images_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])

    # Limit dataset size (optional)
    if max_samples is not None:
        images = images[:max_samples]

    # Shuffle for better distribution
    random.shuffle(images)

    total = len(images)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_files = images[:train_end]
    val_files = images[train_end:val_end]
 

    def copy_files(file_list, split):
        img_out = os.path.join(dest_dir, "images", split)
        lbl_out = os.path.join(dest_dir, "labels", split)

        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for img in file_list:
            base = os.path.splitext(img)[0]
            label = base + ".txt"

            src_img = os.path.join(src_images_dir, img)
            src_lbl = os.path.join(src_labels_dir, label)

            dst_img = os.path.join(img_out, img)
            dst_lbl = os.path.join(lbl_out, label)

            if os.path.exists(src_img):
                shutil.copy2(src_img, dst_img)

            if os.path.exists(src_lbl):
                shutil.copy2(src_lbl, dst_lbl)

    # Copy splits
    copy_files(train_files, "train")
    copy_files(val_files, "val")
   

    print("\nDataset Split Complete (80/20/0)")
    print(f"Total     : {total}")
    print(f"Train     : {len(train_files)}")
    print(f"Validation: {len(val_files)}")

    
def main():
    split_dataset_with_test(
        src_images_dir="data/data_object_image_2/training/image_2",
        src_labels_dir="data/data_object_label_2/training/label_2",
        dest_dir="src/dataset",
        train_ratio=0.8,
        val_ratio=0.2,
        max_samples=7518,
        seed=42
    )

if __name__ == "__main__":
    main()