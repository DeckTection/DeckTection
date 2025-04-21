import os
import csv
import random
from PIL import Image, ImageEnhance
from tqdm import tqdm
import numpy as np
import pandas as pd

class PerspectiveWarp:
    def __init__(self, max_tilt_ratio=0.05):
        self.max_tilt_ratio = max_tilt_ratio

    def __call__(self, image):
        width, height = image.size
        dx = self.max_tilt_ratio * width
        dy = self.max_tilt_ratio * height

        tl = (random.uniform(-dx, dx), random.uniform(-dy, dy))
        tr = (width + random.uniform(-dx, dx), random.uniform(-dy, dy))
        br = (width + random.uniform(-dx, dx), height + random.uniform(-dy, dy))
        bl = (random.uniform(-dx, dx), height + random.uniform(-dy, dy))

        src = [(0, 0), (width, 0), (width, height), (0, height)]
        dst = [tl, tr, br, bl]

        coeffs = self.find_coeffs(dst, src)
        return image.transform(image.size, Image.PERSPECTIVE, coeffs, resample=Image.BICUBIC)

    def find_coeffs(self, dst, src):
        matrix = []
        for p1, p2 in zip(dst, src):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])
        A = np.array(matrix)
        B = np.array(src).flatten()
        return np.linalg.lstsq(A, B, rcond=None)[0]

def augment(image):
    angle = random.uniform(-15, 15)
    image = image.rotate(angle, resample=Image.BICUBIC, expand=True)

    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(random.uniform(0.8, 1.2))

    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(random.uniform(0.8, 1.2))

    return image

def generate_augmented_tests(card_info_csv="card_info.csv", input_dir="../card_images", output_dir="../test_images", output_csv="test_info.csv", num_augments=2, num_total_tests=100):
    os.makedirs(output_dir, exist_ok=True)
    perspective = PerspectiveWarp()

    card_info = pd.read_csv(card_info_csv)
    results = []

    sampled_rows = [random.choice(card_info.iterrows()) for _ in range(num_total_tests)]
    for _, row in tqdm(sampled_rows, total=num_total_tests, desc="Augmenting cards"):
        image_path = os.path.join(input_dir, row['image_name'])

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Failed to open {row['image_name']}: {e}")
            continue

        for i in range(num_augments):
            aug_img = augment(perspective(image))
            new_filename = f"{os.path.splitext(row['image_name'])[0]}_test{i+1}.jpg"
            aug_img.save(os.path.join(output_dir, new_filename))

            results.append({
                "id": len(results),
                "image_name": new_filename,
                "product_name": row["product_name"],
                "mantle_sku": row["mantle_sku"]
            })

    with open(output_csv, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "image_name", "product_name", "mantle_sku"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Saved {len(results)} test images to '{output_dir}' and CSV to '{output_csv}'")

if __name__ == "__main__":
    generate_augmented_tests(num_augments=2)

