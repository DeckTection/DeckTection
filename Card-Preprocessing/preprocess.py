import cv2
import numpy as np
import os
from utils.square import normalize_to_square
from utils.clahe_utils import apply_clahe

def process_and_display_images(input_dir="images", output_dir="normalized_clahe_output"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Valid image extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    # Get all valid image paths
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
    if not image_files:
        print(f"No valid images found in {input_dir}")
        return

    for filename in image_files:
        image_path = os.path.join(input_dir, filename)
        original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if original_image is None:
            print(f"Could not read image: {image_path}")
            continue

        try:
            # Normalize and apply CLAHE
            normalized = normalize_to_square(original_image)
            clahe_image = apply_clahe(normalized)

            # Resize normalized and CLAHE images to match original's shape
            height, width = original_image.shape[:2]
            normalized_resized = cv2.resize(normalized, (width, height))
            clahe_resized = cv2.resize(clahe_image, (width, height))

            # Stack them side-by-side for visual comparison
            combined = np.hstack((original_image, normalized_resized, clahe_resized))

            # Display window
            cv2.imshow(f"{filename} - Original | Normalized | CLAHE", combined)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Save outputs
            base_name = os.path.splitext(filename)[0]
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_normalized.jpg"), normalized)
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_clahe.jpg"), clahe_image)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    process_and_display_images()
