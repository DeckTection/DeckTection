import os
import random
import csv
import requests
import glob
import math
from PIL import Image, ImageDraw, ImageChops
from io import BytesIO
from tqdm import tqdm
import numpy as np
from bs4 import BeautifulSoup
import base64

# Configuration
NUM_COMPOSITES = 1000  # Number of composite images to generate
IMAGE_SIZE = 640
OUTPUT_DIR = "datasets"
BACKGROUNDS_DIR = "backgrounds"
MIN_CARDS = 1
MAX_CARDS = 5

# Create directories
os.makedirs(f"{OUTPUT_DIR}/yolo/images/train", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/yolo/labels/train", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/sam/images/train", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/sam/masks/train", exist_ok=True)

# Load background images
background_paths = glob.glob(f"{BACKGROUNDS_DIR}/*.jpg") + \
                   glob.glob(f"{BACKGROUNDS_DIR}/*.png") + \
                   glob.glob(f"{BACKGROUNDS_DIR}/*.jpeg")

def find_coeffs(pa, pb):
    """Calculate perspective transform coefficients with stability fixes"""
    matrix = []
    for (x, y), (u, v) in zip(pa, pb):
        matrix.append([x, y, 1, 0, 0, 0, -u*x, -u*y])
        matrix.append([0, 0, 0, x, y, 1, -v*x, -v*y])
    
    A = np.array(matrix, dtype=np.float64)
    B = np.array(pb).flatten().astype(np.float64)
    
    try:
        coeffs = np.linalg.lstsq(A, B, rcond=1e-6)[0]
        return coeffs.tolist()
    except np.linalg.LinAlgError:
        return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

def load_cards(csv_path):
    """Load card data from CSV file"""
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
    """Enhanced transformations with edge protection and aggressive rotations"""
    # More aggressive rotation logic
    if random.random() < 0.25:  # 25% chance for dramatic rotation
        angle = random.choice([-90, 90, -75, 75, -60, 60])
    else:
        angle = random.uniform(-45, 45)
    rotated = card_image.rotate(angle, expand=True, resample=Image.BICUBIC)
    
    # Resize with aspect ratio preservation
    original_aspect = rotated.width / rotated.height
    new_width = random.randint(100, 300)
    new_height = int(new_width / original_aspect)
    resized = rotated.resize((new_width, new_height), Image.LANCZOS)
    
    # Create buffer space for perspective transforms
    buffer_multiplier = 1.5  # Extra space to prevent clipping
    buffered_w = int(new_width * buffer_multiplier)
    buffered_h = int(new_height * buffer_multiplier)
    buffered = Image.new("RGBA", (buffered_w, buffered_h))
    paste_x = (buffered_w - new_width) // 2
    paste_y = (buffered_h - new_height) // 2
    buffered.paste(resized, (paste_x, paste_y))
    
    # Aggressive perspective parameters
    max_shift = 0.25  # Increased perspective distortion
    pa = [
        (paste_x, paste_y),
        (paste_x + new_width, paste_y),
        (paste_x + new_width, paste_y + new_height),
        (paste_x, paste_y + new_height)
    ]
    pb = [
        (paste_x + random.randint(-int(new_width*max_shift), int(new_width*max_shift)),
        paste_y + random.randint(-int(new_height*max_shift), int(new_height*max_shift))),
        (paste_x + new_width + random.randint(-int(new_width*max_shift), int(new_width*max_shift)),
        paste_y + random.randint(-int(new_height*max_shift), int(new_height*max_shift))),
        (paste_x + new_width + random.randint(-int(new_width*max_shift), int(new_width*max_shift)),
        paste_y + new_height + random.randint(-int(new_height*max_shift), int(new_height*max_shift))),
        (paste_x + random.randint(-int(new_width*max_shift), int(new_width*max_shift)),
        paste_y + new_height + random.randint(-int(new_height*max_shift), int(new_height*max_shift)))
    ]
    
    coeffs = find_coeffs(pa, pb)
    
    try:
        transformed = buffered.transform(
            buffered.size, Image.Transform.PERSPECTIVE, coeffs, 
            resample=Image.BICUBIC)
    except Exception:
        transformed = buffered
    
    # Auto-crop to visible content with margin
    bbox = transformed.getbbox()
    if bbox:
        margin = 10  # Safety margin
        crop_box = (
            max(0, bbox[0] - margin),
            max(0, bbox[1] - margin),
            min(transformed.width, bbox[2] + margin),
            min(transformed.height, bbox[3] + margin)
        )
        final = transformed.crop(crop_box)
    else:
        final = transformed
    
    # Resize to original target dimensions
    return final.resize((new_width, new_height), Image.LANCZOS)

def download_image(url):
    """Download image from URL and return as PIL Image or None"""
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
    """Improved Google Images search with better result parsing"""
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
    """Generate one composite image with multiple cards"""
    bg = random.choice(backgrounds).copy()
    annotations = []
    masks = []

    num_cards = random.randint(MIN_CARDS, MAX_CARDS)
    selected_cards = random.sample(cards, num_cards)

    for card in selected_cards:
        img = download_image(card['image_url'])
        if img is None:
            print(f"Pulled image from Google for {card['product_name']}")
            img = fetch_google_image(card['product_name'])
        if not img:
            continue

        try:
            img = img.convert("RGBA")
            transformed = process_card_image(img)
            tw, th = transformed.size

            max_x = IMAGE_SIZE - tw
            max_y = IMAGE_SIZE - th
            if max_x < 0 or max_y < 0:
                continue

            x = random.randint(0, max_x)
            y = random.randint(0, max_y)

            bg.paste(transformed, (x, y), transformed)
            annotations.append({'x': x, 'y': y, 'w': tw, 'h': th})

            # Create exact segmentation mask using alpha channel
            card_mask = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)
            alpha = transformed.getchannel('A')
            
            # Threshold alpha to create binary mask (optional)
            alpha = alpha.point(lambda p: 255 if p > 0 else 0)
            
            card_mask.paste(alpha, (x, y), alpha)
            masks.append(card_mask)

        except Exception as e:
            print(f"Error processing card: {e}")

def load_backgrounds():
    """Load and preprocess all background images"""
    backgrounds = []
    for bg_path in background_paths:
        try:
            bg = Image.open(bg_path).convert('RGB')
            if bg.size != (IMAGE_SIZE, IMAGE_SIZE):
                bg = bg.resize((IMAGE_SIZE, IMAGE_SIZE))
            backgrounds.append(bg)
        except Exception as e:
            print(f"Skipping invalid background: {bg_path}")
    return backgrounds

def create_dataset(csv_path):
    """Main dataset creation function"""
    cards = load_cards(csv_path)
    backgrounds = load_backgrounds()
    
    if not backgrounds:
        raise ValueError("No valid background images found")
    
    for composite_id in tqdm(range(NUM_COMPOSITES)):
        generate_composite(composite_id, cards, backgrounds)

if __name__ == "__main__":
    create_dataset(r"$magic-{table}_202504101030.csv")
    print("Dataset generation complete!")