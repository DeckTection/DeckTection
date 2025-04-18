import cv2
import numpy as np

def normalize_to_square(image_path, output_size=512):
    # Read image as grayscale (assuming mask is 0-255)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found or invalid image path")
    
    # Convert to BGR for compatibility with drawing functions
    img_display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Threshold (assuming mask is 0 and 255)
    _, thresh = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    
    # Find contours (using RETR_EXTERNAL for outermost contour)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found")
    
    # Filter contours by area (remove small noise)
    min_contour_area = img.size * 0.01  # 1% of total pixels
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
    if not valid_contours:
        raise ValueError("No valid contours found")
    
    largest_contour = max(valid_contours, key=cv2.contourArea)
    
    # Approximate contour to quadrilateral
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    if len(approx) != 4:
        raise ValueError(f"Expected quadrilateral, got {len(approx)} points")
    
    # Improved point ordering using convex hull approach
    def order_points(pts):
        pts = pts.reshape(-1, 2)
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # Find convex hull and sort clockwise
        hull = cv2.convexHull(pts, returnPoints=True)
        hull = sorted(hull.reshape(-1, 2), key=lambda p: np.arctan2(p[1], p[0]))
        
        # Find centroid
        centroid = np.mean(hull, axis=0)
        
        # Sort points based on angle from centroid
        angles = [np.arctan2(p[1]-centroid[1], p[0]-centroid[0]) for p in hull]
        hull = [p for _, p in sorted(zip(angles, hull))]
        
        # Take the first 4 points (assuming quadrilateral)
        return np.array(hull[:4], dtype=np.float32)
    
    ordered = order_points(approx)
    
    # Calculate destination points with padding
    width = int(max(
        np.linalg.norm(ordered[0] - ordered[1]),
        np.linalg.norm(ordered[2] - ordered[3])
    ))
    height = int(max(
        np.linalg.norm(ordered[1] - ordered[2]),
        np.linalg.norm(ordered[3] - ordered[0])
    ))
    
    # Ensure valid dimensions
    width = max(width, 1)
    height = max(height, 1)
    
    # Create destination points with aspect ratio preservation
    scale = min(output_size/width, output_size/height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    dst = np.array([
        [0, 0],
        [new_width-1, 0],
        [new_width-1, new_height-1],
        [0, new_height-1]
    ], dtype=np.float32)
    
    # Center in output
    x_offset = (output_size - new_width) // 2
    y_offset = (output_size - new_height) // 2
    dst += np.array([[x_offset, y_offset]], dtype=np.float32)
    
    # Calculate perspective transform
    matrix = cv2.getPerspectiveTransform(ordered, dst)
    
    # Perform the warp using the original color image
    warped = cv2.warpPerspective(
        cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),  # Use color image for warping
        matrix,
        (output_size, output_size),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    
    return warped

# Usage
normalized = normalize_to_square(r"C:\Users\willi\Documents\GitHub\DeckTection\Card Pre-processing\card_4_0.0_0.94.jpg")
cv2.imwrite("normalized_card.jpg", normalized)