import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def order_points(pts):
    """Improved point ordering using convex hull approach"""
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

def process_single_image(image_path, output_size=640):
    """Process a single image to normalized square"""
    try:
        # Read image in color
        color_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if color_img is None:
            raise ValueError(f"Image not found: {image_path}")
        
        # Convert to grayscale for processing
        gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        
        # Threshold (assuming mask is 0 and 255)
        _, thresh = cv2.threshold(gray_img, 1, 255, cv2.THRESH_BINARY)
        
        # Find contours (using RETR_EXTERNAL for outermost contour)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError(f"No contours found in {os.path.basename(image_path)}")
        
        # Filter contours by area (remove small noise)
        min_contour_area = gray_img.size * 0.01  # 1% of total pixels
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
        if not valid_contours:
            raise ValueError(f"No valid contours in {os.path.basename(image_path)}")
        
        largest_contour = max(valid_contours, key=cv2.contourArea)
        
        # Approximate contour to quadrilateral
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        if len(approx) != 4:
            raise ValueError(f"Expected quadrilateral, got {len(approx)} points in {os.path.basename(image_path)}")
        
        ordered = order_points(approx)
        
        # Destination points for perspective transform
        dst = np.array([
            [0, 0],
            [output_size-1, 0],
            [output_size-1, output_size-1],
            [0, output_size-1]
        ], dtype=np.float32)
        
        # Calculate perspective transform
        matrix = cv2.getPerspectiveTransform(ordered, dst)
        
        # Perform the warp using the original color image
        warped = cv2.warpPerspective(
            color_img,
            matrix,
            (output_size, output_size),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))
        
        return warped, None
    
    except Exception as e:
        return None, str(e)

def process_images(input_paths, output_dir="normalized_cards", max_workers=4):
    """Process multiple images with parallel execution"""
    os.makedirs(output_dir, exist_ok=True)
    
    successful = 0
    failed = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for img_path in input_paths:
            output_path = os.path.join(output_dir, os.path.basename(img_path))
            futures.append(executor.submit(process_single_image, img_path))
        
        # Process results with progress bar
        for future, img_path in tqdm(zip(futures, input_paths), total=len(input_paths), desc="Processing images"):
            result, error = future.result()
            output_path = os.path.join(output_dir, os.path.basename(img_path))
            
            if result is not None:
                cv2.imwrite(output_path, result)
                successful += 1
            else:
                failed.append((os.path.basename(img_path), error))
    
    # Print summary
    print(f"\nProcessing complete: {successful} successful, {len(failed)} failed")
    if failed:
        print("\nFailed images:")
        for filename, error in failed:
            print(f"{filename}: {error}")
    
    return successful, failed

def get_image_paths(input_dir):
    """Get all image paths from directory"""
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    return [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
            if f.lower().endswith(valid_extensions)]

if __name__ == "__main__":
    # Example usage - process all images in a directory
    input_directory = "images"
    output_directory = "normalized_cards"
    
    image_paths = get_image_paths(input_directory)
    if not image_paths:
        print(f"No images found in {input_directory}")
    else:
        print(f"Found {len(image_paths)} images to process")
        success_count, failure_list = process_images(image_paths, output_directory)
        
        # Optionally process single image
        # normalized = process_single_image("card_1_0.0_0.97.jpg")[0]
        # cv2.imwrite("normalized_card.jpg", normalized)