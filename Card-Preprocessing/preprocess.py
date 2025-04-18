import cv2
import numpy as np
from utils.square import normalize_to_square
from utils.clahe_utils import apply_clahe

# Dummy test
if __name__ == "__main__":
    image_path = "../Card-Detection/card_1_0.0_0.97.jpg"

    # Read image in color
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if original_image is None:
        raise ValueError("Image not found or invalid image path")

    # Usage
    normalized = normalize_to_square(original_image)

    output_path = "normalized_card.jpg"
    cv2.imwrite(output_path, normalized)

    # Load the image
    image = cv2.imread(output_path)
    
    clahe_image = apply_clahe(normalized)

    # Resize normalized and CLAHE images to match original's shape
    height, width = original_image.shape[:2]
    normalized_resized = cv2.resize(normalized, (width, height))
    clahe_resized = cv2.resize(clahe_image, (width, height))

    # Now you can safely stack them side-by-side
    combined = np.hstack((original_image, normalized_resized, clahe_resized))

    cv2.imshow("Original (Left) vs CLAHE Enhanced (Right)", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
