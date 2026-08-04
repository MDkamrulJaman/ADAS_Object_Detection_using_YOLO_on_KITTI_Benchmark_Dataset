import cv2
import os
import random

CLASS_NAMES = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc"]  

def show_yolo(image_path, label_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        cls = int(parts[0])
        x, y, bw, bh = map(float, parts[1:])

        # YOLO → pixel conversion
        x *= w
        y *= h
        bw *= w
        bh *= h

        x1 = int(x - bw/2)
        y1 = int(y - bh/2)
        x2 = int(x + bw/2)
        y2 = int(y + bh/2)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, CLASS_NAMES[cls], (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cv2.imshow("YOLO CHECK", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def main():
    image_path = "C:/Users/HP/OneDrive/Desktop/ADAS_Object_Detection_using_YOLO_on_KITTI_Benchmark_Dataset/src/dataset/images/val/000021.png"
    label_path = "C:/Users/HP/OneDrive/Desktop/ADAS_Object_Detection_using_YOLO_on_KITTI_Benchmark_Dataset/src/dataset/labels/val/000021.txt"

    show_yolo(image_path, label_path)

if __name__ == "__main__":
    main()