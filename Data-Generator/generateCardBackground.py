import os
import random
import csv
import requests
import glob
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
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

def apply_glare_effect(image):
    """Add INTENSE artificial glare with dramatic highlights"""
    # Only apply glare to 80% of cards (more frequent)
    if random.random() > 0.8:
        return image
    
    width, height = image.size
    overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # DRAMATIC glare parameters
    intensity = random.uniform(0.5, 1.2)  # Much stronger intensity range
    cx = random.uniform(-0.1 * width, 1.1 * width)  # More centered
    cy = random.uniform(-0.1 * height, 1.1 * height)
    radius = random.randint(int(min(width, height)*0.3), int(min(width, height)*1.2))  # Larger radius
    
    # Create MULTIPLE glare spots (30% chance)
    if random.random() < 0.3:
        for _ in range(random.randint(1, 3)):
            sub_intensity = random.uniform(0.3, 0.8)
            sub_cx = cx + random.uniform(-0.2*width, 0.2*width)
            sub_cy = cy + random.uniform(-0.2*height, 0.2*height)
            sub_radius = random.randint(int(radius*0.3), int(radius*0.7))
            
            for i in range(sub_radius, 0, -1):
                alpha = int(255 * sub_intensity * (i/sub_radius)**0.3)  # Sharper falloff
                color = (255, 255, 255, alpha)
                draw.ellipse(
                    (sub_cx - i, sub_cy - i, sub_cx + i, sub_cy + i),
                    fill=color
                )
    
    # Main glare spot (always present)
    for i in range(radius, 0, -1):
        alpha = int(255 * intensity * (i/radius)**0.2)  # Very slow falloff for intense core
        color = random.choice([
            (255, 255, 255, alpha),  # Bright white
            (255, 240, 150, alpha),  # Warm golden
            (180, 220, 255, alpha)   # Cool blue
        ])
        draw.ellipse(
            (cx - i, cy - i, cx + i, cy + i),
            fill=color
        )
    
    # Apply blur - less blur for sharper glare
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=max(1, radius//6)))
    
    # Blend modes for more intense effect
    if random.random() < 0.5:  # 50% chance for additive blending
        return Image.blend(image, overlay, 0.7)
    else:
        return Image.alpha_composite(image, overlay)

def apply_hue_shift(image):
    """Apply random hue shift to card pixels only"""
    if random.random() > 0.6:  # 40% chance to apply hue shift
        return image

    img_array = np.array(image)
    rgb = img_array[..., :3]
    alpha = img_array[..., 3] if img_array.shape[2] == 4 else None

    # Create mask of opaque pixels
    mask = (alpha > 0) if alpha is not None else np.ones(rgb.shape[:2], bool)

    # Convert to HSV (uint8 format)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    # Apply hue shift only to masked pixels
    hue_shift = random.randint(-15, 15)
    h_float = h.astype(np.float32)
    h_float[mask] = (h_float[mask] + hue_shift) % 180.0
    h = np.clip(h_float, 0, 179).astype(np.uint8)

    # Apply saturation adjustment to same pixels
    if random.random() < 0.3:
        sat_shift = random.uniform(0.7, 1.3)
        s_float = s.astype(np.float32)
        s_float[mask] = np.clip(s_float[mask] * sat_shift, 0, 255)
        s = s_float.astype(np.uint8)

    # Merge channels
    hsv_shifted = cv2.merge((h, s, v))
    rgb_shifted = cv2.cvtColor(hsv_shifted, cv2.COLOR_HSV2RGB)

    # Reconstruct image
    if alpha is not None:
        return Image.fromarray(np.dstack((rgb_shifted, alpha)), 'RGBA')
    return Image.fromarray(rgb_shifted, 'RGB')

def apply_lighting_effects(image):
    """Apply brightness/contrast adjustments"""
    if random.random() > 0.5:  # 50% chance to apply
        return image
    
    # Random brightness adjustment
    brightness = random.uniform(0.8, 1.2)
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)
    
    # Random contrast adjustment
    if random.random() < 0.3:
        contrast = random.uniform(0.9, 1.5)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast)
    
    return image

def process_card_image(card_image):
    # Apply all visual effects first
    card_image = apply_glare_effect(card_image)
    card_image = apply_hue_shift(card_image)
    card_image = apply_lighting_effects(card_image)

    original_width, original_height = card_image.size
    
    # Apply rotation
    angle = random.uniform(-179, 179)
    rotated = card_image.rotate(angle, expand=True, resample=Image.BICUBIC)
    
    # Apply resizing
    new_width = random.randint(100, 300)
    scale_factor = new_width / rotated.width
    new_height = int(rotated.height * scale_factor)
    resized = rotated.resize((new_width, new_height), Image.LANCZOS)
    
    # Create canvas for perspective transform
    padding = int(max(new_width, new_height) * 0.3)
    canvas_size = (new_width + padding*2, new_height + padding*2)
    canvas = Image.new('RGBA', canvas_size, (0, 0, 0, 0))
    canvas.paste(resized, (padding, padding))
    
    # Perspective transform points
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
    
    # Crop to visible area
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