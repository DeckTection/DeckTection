import cv2
import numpy as np
from utils.preprocessing import preprocess_card
from utils.extract_corners import extract_card_corners

# Dummy test
if __name__ == "__main__":
    image_path = "sample_card.jpg"
    image = cv2.imread(image_path)

    # mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    corners = extract_card_corners(image)

    processed = preprocess_card(image, corners)

    cv2.imshow("Processed Card", processed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
