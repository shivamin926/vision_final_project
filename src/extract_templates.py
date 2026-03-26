import os
import cv2
import numpy as np
from utils import ensure_dir, clean_mask, largest_contour_mask, save_rgba

RAW_DIR = "data/raw"
TEMPLATE_DIR = "data/templates"

FRUITS = ["apple", "banana", "lime"]


def remove_white_background(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # white background = low saturation + high value
    lower_white = np.array([0, 0, 180], dtype=np.uint8)
    upper_white = np.array([180, 80, 255], dtype=np.uint8)

    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    fruit_mask = cv2.bitwise_not(white_mask)

    fruit_mask = clean_mask(fruit_mask, kernel_size=5)
    fruit_mask = largest_contour_mask(fruit_mask)

    fruit_only = cv2.bitwise_and(image, image, mask=fruit_mask)
    return fruit_only, fruit_mask


def process_fruit_class(fruit_name):
    input_dir = os.path.join(RAW_DIR, fruit_name)
    output_dir = os.path.join(TEMPLATE_DIR, fruit_name)

    ensure_dir(output_dir)

    count = 0

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(input_dir, filename)
        image = cv2.imread(path)

        if image is None:
            continue

        fruit_only, fruit_mask = remove_white_background(image)

        if cv2.countNonZero(fruit_mask) < 300:
            continue

        save_name = f"{fruit_name}_{count:02d}.png"
        save_path = os.path.join(output_dir, save_name)

        save_rgba(fruit_only, fruit_mask, save_path)
        count += 1

    print(f"[INFO] Saved {count} templates for {fruit_name}")


def main():
    ensure_dir(TEMPLATE_DIR)

    for fruit in FRUITS:
        process_fruit_class(fruit)

    print("[DONE] Template extraction finished.")


if __name__ == "__main__":
    main()