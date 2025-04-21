import os
import csv
from PIL import Image
import requests
from io import BytesIO
import base64

def download_image(url):
    """Download an image from a URL or decode a base64 data URL."""
    try:
        if url.startswith("data:image"):
            # Handle base64 encoded image data
            header, data = url.split(",", 1)
            image_format = header.split("/")[1].split(";")[0]
            image_data = base64.b64decode(data)
            return Image.open(BytesIO(image_data)), image_format.upper()
        else:
            # Handle regular URL
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return Image.open(BytesIO(response.content)), None
    except Exception as e:
        print(f"Download failed for {url}: {e}")
    return None, None

def create_card_images_and_csv(csv_paths, output_dir="card_images", output_csv="card_info.csv"):
    # Create output directory for card images
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle both single and multiple CSV files
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]
    
    # Load card data from all CSVs
    cards = []
    for csv_path in csv_paths:
        try:
            with open(csv_path, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    img_url = row['Image']
                    # Extract image ID from URL (same method as original code)
                    img_id = img_url.split('/')[-1].split('.')[0]
                    cards.append({
                        'img_id': img_id,
                        'product_name': row['Product Name'],
                        'image_url': img_url
                    })
        except FileNotFoundError:
            print(f"Warning: CSV file not found - {csv_path}")
            continue
        except Exception as e:
            print(f"Error processing {csv_path}: {e}")
            continue
    
    # Process each card to download and save image
    for card in cards:
        img_url = card['image_url']
        img_id = card['img_id']
        product_name = card['product_name']
        
        image, detected_format = download_image(img_url)
        if image:
            # Determine image format with priority to format from data URL
            if detected_format:
                img_format = detected_format
            else:
                # Fallback to Pillow detected format or JPEG
                img_format = image.format or 'JPEG'
            
            # Handle transparency by checking image mode
            if image.mode in ('RGBA', 'LA'):
                img_format = 'PNG'
            
            # Normalize common formats
            ext = img_format.lower()
            if ext == 'jpeg':
                ext = 'jpg'
            
            # Ensure valid filename
            filename = f"{img_id}.{ext}"
            save_path = os.path.join(output_dir, filename)
            
            try:
                # Convert to RGB for JPEG format to avoid alpha issues
                if img_format == 'JPEG' and image.mode in ('RGBA', 'LA'):
                    image = image.convert('RGB')
                image.save(save_path, format=img_format)
                card['filename'] = filename
            except Exception as e:
                print(f"Error saving image {img_id}: {e}")
                card['filename'] = None
        else:
            card['filename'] = None
    
    # Write metadata CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['id', 'image_name', 'product_name']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for card in cards:
            if card.get('filename'):
                writer.writerow({
                    'id': card['img_id'],
                    'image_name': card['filename'],
                    'product_name': card['product_name']
                })
    
    # Print summary
    success_count = sum(1 for card in cards if card.get('filename'))
    print(f"Successfully processed {success_count}/{len(cards)} cards.")
    print(f"Card images saved to: {os.path.abspath(output_dir)}")
    print(f"Metadata CSV saved to: {os.path.abspath(output_csv)}")

if __name__ == "__main__":
    # Example usage patterns:
    
    # Single CSV file
    # create_card_images_and_csv(r"$magic-{table}_202504101030.csv")
    
    # Multiple CSV files
    create_card_images_and_csv([
        r"$magic-{table}_202504101030.csv",
        r"$Pokemon-{table}_202504112100.csv",
        r"$YuGiOh-{table}_202504112056.csv"
    ])
    
    # Or using glob pattern
    # import glob
    # create_card_images_and_csv(glob.glob("csv_files/*.csv"))
    
    print("Process completed!")