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
from bs4 import BeautifulSoup

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

def simple_download_image(url):
    """Download an image from a URL or decode a base64 data URL, returning the Image object or None."""
    try:
        if url.startswith("data:image"):
            header, data = url.split(",", 1)
            image_data = base64.b64decode(data)
            return Image.open(BytesIO(image_data))
            
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"Download failed: {e}")
    return None

def fetch_google_image(product_name):
    """Fetch product image from Google using site:scryfall.com search."""
    try:
        query = f"{product_name} site:scryfall.com"
        search_url = "https://www.google.com/search"
        params = {"q": query, "tbm": "isch"}
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}

        response = requests.get(search_url, headers=headers, params=params, timeout=15)
        soup = BeautifulSoup(response.text, "html.parser")
        
        images = []
        for img in soup.find_all("img"):
            src = img.get("src")
            if src and ("http" in src or "data:image" in src):
                images.append(src)
        
        # Try first 3 results to find a valid image
        for img_url in images[:3]:
            img = simple_download_image(img_url)
            if img and img.width > 100:
                return img
    except Exception as e:
        print(f"Google search failed for '{product_name}': {e}")
    return None

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

    # Attempt to download original image
    image, detected_format = download_image(img_url)
    
    # If download failed, try Google image search
    if image is None:
        image = fetch_google_image(product_name)
        detected_format = None  # Will be determined from image data if available
    
    if image:
        try:
            # Convert to RGBA if necessary
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
            
            # Determine image format
            img_format = detected_format or image.format or 'JPEG'
            ext = 'png'  # Force PNG for consistency
            
            filename = f"{img_id}.{ext}"
            save_path = os.path.join(output_dir, filename)

            # Resize and save original
            image_resized = image.resize((640, 640), Image.Resampling.LANCZOS)
            image_resized.save(save_path, format='PNG')

            # Apply perspective warp
            perspective_warp = PerspectiveWarp(probability=1.0)
            warped_image = preprocess(perspective_warp(image_resized))

            # Save warped image
            warped_filename = f"{img_id}_warped.{ext}"
            warped_save_path = os.path.join(output_dir, warped_filename)
            warped_image.save(warped_save_path, format='PNG')

            return [
                {'id': img_id, 'image_name': filename, 'product_name': product_name, 'mantle_sku': mantle_sku},
                {'id': img_id, 'image_name': warped_filename, 'product_name': product_name, 'mantle_sku': mantle_sku}
            ]
        except Exception as e:
            print(f"Error processing {img_id}: {e}")
    return None

def create_card_images_and_csv(csv_paths, output_dir="../card_images", output_csv="card_info.csv", limit=100):
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]

    # Load existing IDs from output_csv to skip duplicates
    existing_ids = set()
    if os.path.exists(output_csv):
        with open(output_csv, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            existing_ids = {row['id'] for row in reader}

    cards = []
    for csv_path in csv_paths:
        try:
            with open(csv_path, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in itertools.islice(reader, limit):
                    img_url = row['Image']
                    img_id = img_url.split('/')[-1].split('.')[0]

                    if img_id in existing_ids:
                        continue  # Skip if already processed

                    # Build new product_name
                    card_type = row.get('Card Type', '').strip()
                    card_number = row.get('Card Number', '').strip()
                    foil = row.get('Foil', '').strip()
                    product_name = row.get('Product Name', '').strip()

                    product_name = f"{product_name} {card_type} {card_number} {foil}".strip()

                    mantle_sku = row['Mantle SKU']
                    cards.append({
                        'img_id': img_id,
                        'product_name': product_name,
                        'image_url': img_url,
                        'mantle_sku': mantle_sku
                    })
        except FileNotFoundError:
            print(f"Warning: CSV file not found - {csv_path}")
            continue
        except Exception as e:
            print(f"Error processing {csv_path}: {e}")
            continue

    with open("temp_cards.csv", 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['img_id', 'image_url', 'product_name', 'mantle_sku']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for card in cards:
            writer.writerow(card)

    print(f"Prepared {len(cards)} new cards from CSV(s) (skipping {len(existing_ids)} already processed).")

    # Step 2: Load the limited cards
    cards = []
    with open("temp_cards.csv", newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        cards = [row for row in reader]

    # Step 3: Process and save images
    results = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(process_card, card, output_dir) for card in cards]
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.extend(result)  # Flatten the list of results

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['id', 'image_name', 'product_name', 'mantle_sku']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

if __name__ == "__main__":
    # Example usage:
    # Single CSV with 1000 cards
    create_card_images_and_csv([
        r"$Pokemon-{table}_202504112100.csv"
    ], limit=1000)

    # Multiple CSVs with 100 cards each
    # create_card_images_and_csv([
    #     r"$magic-{table}_202504101030.csv",
    #     r"$Pokemon-{table}_202504112100.csv",
    #     r"$YuGiOh-{table}_202504112056.csv"
    # ], limit=100)