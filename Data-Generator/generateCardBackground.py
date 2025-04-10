import os
import random
import requests
import glob
from PIL import Image, ImageDraw
from io import BytesIO
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split

# Configuration
NUM_IMAGES = 1000  # Adjust based on needs
IMAGE_SIZE = 640
OUTPUT_DIR = "datasets"
BACKGROUNDS_DIR = "backgrounds"  # Directory with background images
THREADS = 8  # Be careful with rate limiting

# Create directories
os.makedirs(f"{OUTPUT_DIR}/yolo/images/train", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/yolo/labels/train", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/sam/images/train", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/sam/masks/train", exist_ok=True)

# Load background images
background_paths = glob.glob(f"{BACKGROUNDS_DIR}/*.jpg") + \
                   glob.glob(f"{BACKGROUNDS_DIR}/*.png") + \
                   glob.glob(f"{BACKGROUNDS_DIR}/*.jpeg")

if not background_paths:
    raise ValueError(f"No background images found in {BACKGROUNDS_DIR}")

def load_background():
    """Load and preprocess a random background image"""
    bg_path = random.choice(background_paths)
    try:
        bg = Image.open(bg_path).convert('RGB')
        # Resize or crop background to target size
        if bg.size != (IMAGE_SIZE, IMAGE_SIZE):
            if min(bg.size) < IMAGE_SIZE:
                bg = bg.resize((IMAGE_SIZE, IMAGE_SIZE))
            else:
                # Random crop
                x = random.randint(0, bg.width - IMAGE_SIZE)
                y = random.randint(0, bg.height - IMAGE_SIZE)
                bg = bg.crop((x, y, x+IMAGE_SIZE, y+IMAGE_SIZE))
        return bg
    except Exception as e:
        print(f"Error loading background {bg_path}: {e}")
        return load_background()  # Try another background

def download_image(img_id):
    """Download image from CDN with error handling"""
    url = f"https://cdn.mantlestores.com/magic/{img_id}.jpeg"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
    except Exception as e:
        return None

def process_image(img_id):
    # Download image
    fg_image = download_image(img_id)
    if not fg_image:
        return
    
    # Convert to RGBA and resize
    fg_image = fg_image.convert("RGBA")
    fg_width = random.randint(100, 500)
    fg_height = random.randint(100, 500)
    fg_image = fg_image.resize((fg_width, fg_height))
    
    # Load and prepare background
    bg_image = load_background()
    
    # Random position with boundary checking
    max_x = bg_image.width - fg_width
    max_y = bg_image.height - fg_height
    if max_x < 0 or max_y < 0:
        return  # Skip images larger than background
    
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)
    
    # Create composite
    composite = bg_image.copy()
    composite.paste(fg_image, (x, y), fg_image)
    
    # Save datasets
    save_yolo_data(composite, img_id, x, y, fg_width, fg_height)
    save_sam_data(composite, img_id, x, y, fg_width, fg_height)

def save_yolo_data(image, img_id, x, y, w, h):
    # Normalize coordinates
    x_center = (x + w/2) / IMAGE_SIZE
    y_center = (y + h/2) / IMAGE_SIZE
    width = w / IMAGE_SIZE
    height = h / IMAGE_SIZE
    
    # Save image
    image.save(f"{OUTPUT_DIR}/yolo/images/train/{img_id}.jpg")
    
    # Save label
    with open(f"{OUTPUT_DIR}/yolo/labels/train/{img_id}.txt", "w") as f:
        f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

def save_sam_data(image, img_id, x, y, w, h):
    # Save image
    image.save(f"{OUTPUT_DIR}/sam/images/train/{img_id}.jpg")
    
    # Create mask
    mask = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([x, y, x+w, y+h], fill=255)
    mask.save(f"{OUTPUT_DIR}/sam/masks/train/{img_id}.png")

def create_dataset():
    # Generate random image IDs
    image_ids = random.sample(range(1, 300000), NUM_IMAGES)
    
    # Process images with threading
    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        list(tqdm(executor.map(process_image, image_ids), total=NUM_IMAGES))
    
    # Create validation split
    create_validation_split()

def create_validation_split():
    # For YOLO
    images = [f for f in os.listdir(f"{OUTPUT_DIR}/yolo/images/train")]
    train, val = train_test_split(images, test_size=0.1)
    
    for img in val:
        os.rename(f"{OUTPUT_DIR}/yolo/images/train/{img}", 
                 f"{OUTPUT_DIR}/yolo/images/val/{img}")
        label = img.split(".")[0] + ".txt"
        os.rename(f"{OUTPUT_DIR}/yolo/labels/train/{label}", 
                 f"{OUTPUT_DIR}/yolo/labels/val/{label}")
    
    # For SAM
    images = [f for f in os.listdir(f"{OUTPUT_DIR}/sam/images/train")]
    train, val = train_test_split(images, test_size=0.1)
    
    for img in val:
        os.rename(f"{OUTPUT_DIR}/sam/images/train/{img}", 
                 f"{OUTPUT_DIR}/sam/images/val/{img}")
        mask = img.split(".")[0] + ".png"
        os.rename(f"{OUTPUT_DIR}/sam/masks/train/{mask}", 
                 f"{OUTPUT_DIR}/sam/masks/val/{mask}")

if __name__ == "__main__":
    create_dataset()
    print("Dataset creation complete!")