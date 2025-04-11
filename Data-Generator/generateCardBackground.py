import os
import random
import csv
import requests
import glob
from PIL import Image, ImageDraw, ImageChops
from io import BytesIO
from tqdm import tqdm
import numpy as np
from bs4 import BeautifulSoup
import base64
import cv2 

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

def process_card_image(card_image, return_corners=False):
    """Enhanced transformations with corner tracking"""
    original_width, original_height = card_image.size
    
    # Track original corners (top-left, top-right, bottom-right, bottom-left)
    orig_corners = np.array([
        [0, 0],
        [original_width, 0],
        [original_width, original_height],
        [0, original_height]
    ], dtype=np.float32).reshape(-1, 1, 2)  # Reshape for OpenCV compatibility
    
    # Apply rotation
    angle = random.uniform(-45, 45)
    rotated = card_image.rotate(angle, expand=True, resample=Image.BICUBIC)
    
    # Calculate rotation matrix and transform corners
    rotation_matrix = cv2.getRotationMatrix2D(
        (original_width/2, original_height/2), angle, 1)
    rotated_corners = cv2.transform(
        orig_corners, rotation_matrix).squeeze()
    
    # Calculate new bounding box after rotation
    rotated_width = rotated.width
    rotated_height = rotated.height
    
    # Apply resizing
    new_width = random.randint(100, 300)
    scale_factor = new_width / rotated_width
    new_height = int(rotated_height * scale_factor)
    resized = rotated.resize((new_width, new_height), Image.LANCZOS)
    
    # Scale corners
    scaled_corners = rotated_corners * scale_factor
    
    # Apply perspective transform
    buffer_multiplier = 1.5
    buffered_w = int(new_width * buffer_multiplier)
    buffered_h = int(new_height * buffer_multiplier)
    
    # Create perspective transform points (ensure proper shape)
    src_points = np.array([
        [buffered_w/2 - new_width/2, buffered_h/2 - new_height/2],
        [buffered_w/2 + new_width/2, buffered_h/2 - new_height/2],
        [buffered_w/2 + new_width/2, buffered_h/2 + new_height/2],
        [buffered_w/2 - new_width/2, buffered_h/2 + new_height/2]
    ], dtype=np.float32).reshape(-1, 1, 2)  # Reshape for OpenCV
    
    max_shift = 0.25
    dst_points = src_points + np.random.uniform(
        -new_width*max_shift, new_width*max_shift, size=(4, 1, 2))
    
    # Ensure we have exactly 4 points
    if len(src_points) != 4 or len(dst_points) != 4:
        if return_corners:
            return resized, [[0, 0], [new_width, 0], [new_width, new_height], [0, new_height]]
        return resized
    
    # Calculate perspective matrix
    perspective_matrix = cv2.getPerspectiveTransform(
        src_points.astype(np.float32), 
        dst_points.astype(np.float32))
    
    # Transform scaled corners through perspective
    perspective_corners = cv2.perspectiveTransform(
        scaled_corners.reshape(1, -1, 2), perspective_matrix).squeeze()
    
    # Transform to final image coordinates
    final_corners = perspective_corners + [
        buffered_w/2 - new_width/2,
        buffered_h/2 - new_height/2
    ]
    
    # Apply the perspective transform to the image
    try:
        transformed = Image.new("RGBA", (buffered_w, buffered_h))
        warped = resized.transform(
            (buffered_w, buffered_h),
            Image.Transform.PERSPECTIVE,
            perspective_matrix.flatten()[:8],  # PIL wants 8 values
            resample=Image.BICUBIC
        )
        transformed.paste(warped, (0, 0))
    except Exception:
        transformed = resized
    
    # Auto-crop and calculate final corners
    bbox = transformed.getbbox()
    if bbox:
        crop_x1, crop_y1, crop_x2, crop_y2 = bbox
        transformed = transformed.crop(bbox)
        # Adjust corners for cropping
        final_corners -= [crop_x1, crop_y1]
    else:
        crop_x1 = crop_y1 = 0
    
    # Resize to original target dimensions
    final_width, final_height = new_width, new_height
    transformed = transformed.resize((final_width, final_height), Image.LANCZOS)
    
    # Scale corners to final image size
    scale_x = final_width / (crop_x2 - crop_x1) if bbox else 1
    scale_y = final_height / (crop_y2 - crop_y1) if bbox else 1
    final_corners *= [scale_x, scale_y]
    
    if return_corners:
        return transformed, final_corners.astype(int).tolist()
    return transformed

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
            img = fetch_google_image(card['product_name'])
        if not img:
            continue

        try:
            img = img.convert("RGBA")
            transformed, corners = process_card_image(img, return_corners=True)
            tw, th = transformed.size

            max_x = IMAGE_SIZE - tw
            max_y = IMAGE_SIZE - th
            if max_x < 0 or max_y < 0:
                continue

            x = random.randint(0, max_x)
            y = random.randint(0, max_y)

            # Verify we got valid corners
            if len(corners) != 4:
                corners = [
                    [x, y],
                    [x + tw, y],
                    [x + tw, y + th],
                    [x, y + th]
                ]
            else:
                # Adjust corners for final position
                corners = [(x + px, y + py) for (px, py) in corners]
            
            bg.paste(transformed, (x, y), transformed)
            annotations.append(corners)

            # Create precise mask
            mask = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)
            draw = ImageDraw.Draw(mask)
            draw.polygon(corners, fill=255)
            masks.append(mask)

        except Exception as e:
            print(f"Error processing card: {e}")

    # Rest of the function remains the same...

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