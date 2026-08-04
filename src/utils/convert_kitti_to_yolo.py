from pathlib import Path
import shutil


LABELS_DIR = Path("C:/Users/HP/OneDrive/Desktop/ADAS_Object_Detection_using_YOLO_on_KITTI_Benchmark_Dataset/src/dataset/labels")
IMG_WIDTH  = 1242
IMG_HEIGHT = 375
CLASS_MAP = {
    "Car": 0,
    "Van": 1,
    "Truck": 2,
    "Pedestrian": 3,
    "Person_sitting": 4,
    "Cyclist": 5,
    "Tram": 6,
    "Misc": 7,
}

SKIP_CLASSES = {"DontCare"}


def convert_file(kitti_path: Path):
    yolo_lines = []

    with open(kitti_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue

            class_name = parts[0]

            if class_name in SKIP_CLASSES:
                continue
            if class_name not in CLASS_MAP:
                continue

            class_id = CLASS_MAP[class_name]

            x_min = float(parts[4])
            y_min = float(parts[5])
            x_max = float(parts[6])
            y_max = float(parts[7])

            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(IMG_WIDTH, x_max)
            y_max = min(IMG_HEIGHT, y_max)

            x_center = (x_min + x_max) / 2 / IMG_WIDTH
            y_center = (y_min + y_max) / 2 / IMG_HEIGHT
            box_width = (x_max - x_min) / IMG_WIDTH
            box_height = (y_max - y_min) / IMG_HEIGHT

            yolo_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
            )

    with open(kitti_path, "w") as f:
        f.write("\n".join(yolo_lines))


def process_split(split):
    split_dir = LABELS_DIR / split

    if not split_dir.exists():
        print(f"Skipping {split} — not found")
        return 0, 0

    txt_files = sorted(split_dir.glob("*.txt"))
    print(f"\n[{split}] Converting {len(txt_files)} files...")

    total_boxes = 0

    for txt_file in txt_files:
        convert_file(txt_file)

        with open(txt_file) as f:
            total_boxes += len([l for l in f.readlines() if l.strip()])

    print(f"[{split}] Done")
    return len(txt_files), total_boxes


def main():
    splits = ["train", "val"]

    total_files = 0
    total_boxes = 0

    for split in splits:
        files, boxes = process_split(split)
        total_files += files
        total_boxes += boxes

    print(f"\nConversion complete!")
    print(f"Files converted : {total_files}")
    print(f"Total boxes     : {total_boxes}")

if __name__ == "__main__":
    main()