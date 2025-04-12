import os
import random
import csv
import requests
import glob
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import numpy as np
from bs4 import BeautifulSoup
import base64
import cv2 

# Configuration
NUM_COMPOSITES = 1000
IMAGE_SIZE = 640
OUTPUT_DIR = "datasets"
BACKGROUNDS_DIR = "backgrounds"
MIN_CARDS = 1
MAX_CARDS = 5

# Create directories
os.makedirs(f"{OUTPUT_DIR}/yolo/images/train", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/yolo/labels/train", exist_ok=True)

# Load background images
background_paths = glob.glob(f"{BACKGROUNDS_DIR}/*.jpg") + \
                   glob.glob(f"{BACKGROUNDS_DIR}/*.png") + \
                   glob.glob(f"{BACKGROUNDS_DIR}/*.jpeg")

def load_cards(csv_path):
    cards = []
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            img_url = row['Image']
            img_id = img_url.split('/')[-1].split('.')[0]
            cards.append({
                'img_id': img_id,
                'product_name': row['Product Name'],
                'image_url': img_url
            })
    return cards

def process_card_image(card_image):
    original_width, original_height = card_image.size
    
    # Apply rotation
    angle = random.uniform(-179, 179)
    rotated = card_image.rotate(angle, expand=True, resample=Image.BICUBIC)
    
    # Apply resizing
    new_width = random.randint(100, 300)
    scale_factor = new_width / rotated.width
    new_height = int(rotated.height * scale_factor)
    resized = rotated.resize((new_width, new_height), Image.LANCZOS)
    
    # Create a larger canvas to contain the transformed card without cropping
    padding = int(max(new_width, new_height) * 0.3)  # Extra space for transformation
    canvas_size = (new_width + padding*2, new_height + padding*2)
    canvas = Image.new('RGBA', canvas_size, (0, 0, 0, 0))
    canvas.paste(resized, (padding, padding))
    
    # Apply perspective transform
    src_points = np.array([
        [padding, padding],
        [padding + new_width, padding],
        [padding + new_width, padding + new_height],
        [padding, padding + new_height]
    ], dtype=np.float32)

    perspective_mode = random.choice(['axis_rotation', 'random_warp', 'combined'])
    max_shift = 0.2

    if perspective_mode == 'axis_rotation':
        tilt_type = random.choice(['horizontal', 'vertical'])
        base_shift = random.uniform(0.1, 0.25)
        noise_scale = random.uniform(0.05, 0.15)

        if tilt_type == 'horizontal':
            dst_points = np.array([
                [padding + new_width*base_shift, padding - new_height*noise_scale],
                [padding + new_width*(1-base_shift), padding - new_height*noise_scale],
                [padding + new_width*(1+base_shift*0.5), padding + new_height*(1+noise_scale)],
                [padding - new_width*base_shift*0.5, padding + new_height*(1+noise_scale)]
            ], dtype=np.float32)
        else:
            dst_points = np.array([
                [padding - new_width*noise_scale, padding + new_height*base_shift],
                [padding + new_width*(1+noise_scale), padding + new_height*base_shift],
                [padding + new_width*(1+noise_scale*0.5), padding + new_height*(1-base_shift)],
                [padding - new_width*noise_scale*0.5, padding + new_height*(1-base_shift)]
            ], dtype=np.float32)

    elif perspective_mode == 'random_warp':
        dst_points = src_points + np.random.uniform(
            -new_width*max_shift, new_width*max_shift, size=(4, 2))
    else:
        base_shift = random.uniform(0.1, 0.2)
        dst_points = np.array([
            [padding + base_shift*new_width, padding - base_shift*new_height],
            [padding + new_width*(1-base_shift), padding - base_shift*new_height],
            [padding + new_width*(1+base_shift), padding + new_height*(1+base_shift)],
            [padding - base_shift*new_width, padding + new_height*(1+base_shift)]
        ], dtype=np.float32)
        dst_points += np.random.uniform(-new_width*0.1, new_width*0.1, size=(4, 2))

    try:
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        transformed = canvas.transform(
            canvas_size,
            Image.Transform.PERSPECTIVE,
            matrix.flatten()[:8],
            resample=Image.BICUBIC
        )
    except Exception:
        transformed = canvas
    
    # Get bounding box of non-transparent pixels
    bbox = transformed.getbbox()
    if bbox:
        transformed = transformed.crop(bbox)
    
    return transformed

def download_image(url):
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
    try:
        query = f"{product_name} site:scryfall.com"
        search_url = "https://www.google.com/search"
        params = {"q": query, "tbm": "isch"}
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}

        response = requests.get(search_url, headers=headers, params=params)
        soup = BeautifulSoup(response.text, "html.parser")
        
        images = []
        for img in soup.find_all("img"):
            src = img.get("src")
            if src and ("http" in src or "data:image" in src):
                images.append(src)
        
        for img_url in images[:3]:
            img = download_image(img_url)
            if img and img.width > 100:
                return img
    except Exception as e:
        print(f"Google search failed: {e}")
    return None

def generate_composite(composite_id, cards, backgrounds):
    bg = random.choice(backgrounds).copy().resize((IMAGE_SIZE, IMAGE_SIZE))
    annotations = []
    num_cards = random.randint(MIN_CARDS, MAX_CARDS)
    selected_cards = random.sample(cards, num_cards)

    for card in selected_cards:
        img = download_image(card['image_url']) or fetch_google_image(card['product_name'])
        if not img:
            continue

        try:
            img = img.convert("RGBA")
            transformed = process_card_image(img)
            if not transformed:
                continue
                
            tw, th = transformed.size
            
            # Calculate maximum allowed position to keep full card visible
            max_x = IMAGE_SIZE - tw
            max_y = IMAGE_SIZE - th
            if max_x < 0 or max_y < 0:
                continue  # Skip if card is larger than image

            # Try to find non-overlapping position
            max_attempts = 10
            placed = False
            for _ in range(max_attempts):
                x = random.randint(0, max_x)
                y = random.randint(0, max_y)
                new_bbox = (x, y, x + tw, y + th)
                
                # Check for complete overlap with existing cards
                complete_overlap = False
                for existing_bbox in annotations:
                    # Check if new is completely inside existing
                    inside_existing = (new_bbox[0] >= existing_bbox[0] and
                                      new_bbox[1] >= existing_bbox[1] and
                                      new_bbox[2] <= existing_bbox[2] and
                                      new_bbox[3] <= existing_bbox[3])
                    # Check if existing is completely inside new
                    existing_inside_new = (existing_bbox[0] >= new_bbox[0] and
                                          existing_bbox[1] >= new_bbox[1] and
                                          existing_bbox[2] <= new_bbox[2] and
                                          existing_bbox[3] <= new_bbox[3])
                    
                    if inside_existing or existing_inside_new:
                        complete_overlap = True
                        break
                
                if not complete_overlap:
                    placed = True
                    break

            if not placed:
                continue  # Skip this card after max attempts
            
            # Create axis-aligned bounding box
            bbox = new_bbox
            
            # Paste card onto background
            bg.paste(transformed, (x, y), transformed)
            annotations.append(bbox)

        except Exception as e:
            print(f"Error processing card: {e}")

    if annotations:
        # Save image
        bg.convert('RGB').save(f"{OUTPUT_DIR}/yolo/images/train/{composite_id}.jpg")
        
        # Save YOLO annotations
        with open(f"{OUTPUT_DIR}/yolo/labels/train/{composite_id}.txt", "w") as f:
            for x_min, y_min, x_max, y_max in annotations:
                # Calculate normalized center coordinates and dimensions
                width = x_max - x_min
                height = y_max - y_min
                cx = (x_min + x_max) / 2 / IMAGE_SIZE
                cy = (y_min + y_max) / 2 / IMAGE_SIZE
                w = width / IMAGE_SIZE
                h = height / IMAGE_SIZE
                
                # Ensure values are within valid range [0, 1]
                cx = max(0.0, min(cx, 1.0))
                cy = max(0.0, min(cy, 1.0))
                w = max(0.0, min(w, 1.0))
                h = max(0.0, min(h, 1.0))
                
                # Skip boxes that are too small
                if w * h < (100 / (IMAGE_SIZE**2)):
                    continue
                
                f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

def load_backgrounds():
    backgrounds = []
    for bg_path in background_paths:
        try:
            bg = Image.open(bg_path).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
            backgrounds.append(bg)
        except Exception as e:
            print(f"Skipping invalid background: {bg_path}")
    return backgrounds

def create_dataset(csv_path):
    cards = load_cards(csv_path)
    backgrounds = load_backgrounds()
    
    if not backgrounds:
        raise ValueError("No valid background images found")
    
    for composite_id in tqdm(range(NUM_COMPOSITES)):
        generate_composite(composite_id, cards, backgrounds)

if __name__ == "__main__":
    create_dataset(r"$magic-{table}_202504101030.csv")
    print("Dataset generation complete!")