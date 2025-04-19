import os
import csv
from PIL import Image
import requests
from io import BytesIO
import base64
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import random
import numpy as np
import sys

# Add the root of your project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now you can do a normal import
from utils.cardDatasetUtils import load_card_dataset, save_dataset
from card_preprocessing.preprocess import preprocess

# Helper function to apply perspective warp
class PerspectiveWarp:
    def __init__(self, max_tilt_ratio=0.05, probability=0.5):
        self.max_tilt_ratio = max_tilt_ratio  # ~5% of width/height shift
        self.probability = probability

    def __call__(self, image):
        if random.random() > self.probability:
            return image

        width, height = image.size
        dx = self.max_tilt_ratio * width
        dy = self.max_tilt_ratio * height

        # Simulate natural camera tilt: slight inward or outward movement
        # Imagine nudging the top/bottom or left/right corners slightly
        tl = (random.uniform(-dx, dx), random.uniform(-dy, dy))  # top-left
        tr = (width + random.uniform(-dx, dx), random.uniform(-dy, dy))  # top-right
        br = (width + random.uniform(-dx, dx), height + random.uniform(-dy, dy))  # bottom-right
        bl = (random.uniform(-dx, dx), height + random.uniform(-dy, dy))  # bottom-left

        src = [(0, 0), (width, 0), (width, height), (0, height)]
        dst = [tl, tr, br, bl]

        coeffs = self.find_coeffs(dst, src)
        return image.transform(image.size, Image.PERSPECTIVE, coeffs, resample=Image.BICUBIC)

    def find_coeffs(self, dst, src):
        """Compute perspective transform matrix from src to dst."""
        matrix = []
        for p1, p2 in zip(dst, src):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])
        A = np.array(matrix)
        B = np.array(src).flatten()
        return np.linalg.lstsq(A, B, rcond=None)[0]

def download_image(url):
    """Download an image from a URL or decode a base64 data URL."""
    try:
        if url.startswith("data:image"):
            header, data = url.split(",", 1)
            image_format = header.split("/")[1].split(";")[0]
            image_data = base64.b64decode(data)
            return Image.open(BytesIO(image_data)), image_format.upper()
        else:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return Image.open(BytesIO(response.content)), None
    except Exception as e:
        print(f"Download failed for {url}: {e}")
    return None, None


def process_card(card, output_dir):
    img_url = card['image_url']
    img_id = card['img_id']
    product_name = card['product_name']
    mantle_sku = card['mantle_sku']

    # Download original image
    image, detected_format = download_image(img_url)
    if image:
        # Save original image
        img_format = detected_format or image.format or 'JPEG'
        if image.mode in ('RGBA', 'LA'):
            img_format = 'PNG'
        ext = img_format.lower()
        if ext == 'jpeg':
            ext = 'jpg'

        filename = f"{img_id}.{ext}"
        save_path = os.path.join(output_dir, filename)

        try:
            if img_format == 'JPEG' and image.mode in ('RGBA', 'LA'):
                image = image.convert('RGB')
            image = preprocess(image)
            image.save(save_path, format=img_format)

            # Apply perspective warp and save
            perspective_warp = PerspectiveWarp(probability=1.0)  # 100% chance for the warp
            warped_image = preprocess(perspective_warp(image))
            

            # import matplotlib.pyplot as plt
            # # Display original and warped images side by side
            # plt.figure(figsize=(8, 4))
            # plt.subplot(1, 2, 1)
            # plt.imshow(image)
            # plt.title("Original")
            # plt.axis('off')

            # plt.subplot(1, 2, 2)
            # plt.imshow(warped_image)
            # plt.title("Warped")
            # plt.axis('off')

            # plt.tight_layout()
            # plt.show()
            # # import pdb;pdb.set_trace()
            # Save warped image with a "_warped" suffix
            warped_filename = f"{img_id}_warped.{ext}"
            warped_save_path = os.path.join(output_dir, warped_filename)
            warped_image.save(warped_save_path, format=img_format)

            # Return information for both the original and warped images (each as separate entries)
            return [
                {
                    'id': img_id,
                    'image_name': filename,
                    'product_name': product_name,
                    'mantle_sku': mantle_sku  # Original image
                },
                {
                    'id': img_id,
                    'image_name': warped_filename,
                    'product_name': product_name,
                    'mantle_sku': mantle_sku  # Warped image (same label)
                }
            ]
        except Exception as e:
            print(f"Error saving image {img_id}: {e}")
    return None


def create_card_images_and_csv(csv_paths, output_dir="../card_images", output_csv="card_info.csv", limit=100):
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]

    cards = []
    for csv_path in csv_paths:
        try:
            with open(csv_path, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in itertools.islice(reader, limit):  # limit rows
                    img_url = row['Image']
                    img_id = img_url.split('/')[-1].split('.')[0]
                    product_name = row['Product Name']
                    mantle_sku = row['Mantle SKU']  # Get the Mantle SKU from the CSV
                    cards.append({
                        'img_id': img_id,
                        'product_name': product_name,
                        'image_url': img_url,
                        'mantle_sku': mantle_sku  # Add Mantle SKU to the card
                    })
        except FileNotFoundError:
            print(f"Warning: CSV file not found - {csv_path}")
            continue
        except Exception as e:
            print(f"Error processing {csv_path}: {e}")
            continue

    # Write card list to a temp file so it can be picked up by main block
    with open("temp_cards.csv", 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['img_id', 'image_url', 'product_name', 'mantle_sku']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for card in cards:
            writer.writerow(card)

    print(f"Prepared {len(cards)} cards from CSV(s).")


if __name__ == "__main__":
    # # Step 1: Extract top 100 per CSV
    # create_card_images_and_csv([
    #     r"$magic-{table}_202504101030.csv",
    #     r"$Pokemon-{table}_202504112100.csv",
    #     r"$YuGiOh-{table}_202504112056.csv"
    # ], limit=100)

    # Step 1: Extract top 100 per CSV
    create_card_images_and_csv([
        r"$magic-{table}_202504101030.csv"
    ], limit=6000)

    # Step 2: Load the limited cards
    cards = []
    with open("temp_cards.csv", newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        cards = [row for row in reader]

    # Step 3: Process and save images
    output_dir = '../card_images'
    output_csv = 'card_info.csv'
    results = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(process_card, card, output_dir) for card in cards]
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.extend(result)  # Flatten the list of results (because each process returns 2 entries)

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['id', 'image_name', 'product_name', 'mantle_sku']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
