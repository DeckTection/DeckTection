import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from scipy import ndimage

def clean_mask(mask):
    """Post-process mask to remove small artifacts and keep only the main object"""
    # Keep largest connected component
    labeled_mask, num_labels = ndimage.label(mask)
    if num_labels == 0:
        return mask
    
    largest_component = np.zeros_like(mask)
    max_size = 0
    for label in range(1, num_labels + 1):
        component = (labeled_mask == label).astype(np.uint8)
        size = np.sum(component)
        if size > max_size:
            max_size = size
            largest_component = component
    
    # Fill small holes
    filled_mask = ndimage.binary_fill_holes(largest_component).astype(np.uint8)
    
    # Smooth edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    smoothed_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_CLOSE, kernel)
    
    return smoothed_mask

def process_image(image_path, output_path='images/output.jpg'):
    # Create output directories
    os.makedirs('images', exist_ok=True)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Load YOLO model
    model = YOLO('runs/detect/train2/weights/best.pt')
    
    # Initialize SAM
    sam_checkpoint = "sam2-main/checkpoints/sam2.1_hiera_base_plus.pt"
    sam_config = r"C:\Users\willi\Documents\GitHub\DeckTection\Card-Detection\sam2-main\sam2\configs\sam2.1\sam2.1_hiera_b+.yaml"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam_model = build_sam2(sam_config, sam_checkpoint)
    sam_model.to(device)
    sam_predictor = SAM2ImagePredictor(sam_model)
    
    # Read input image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    # Convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # YOLO detection
    results = model.predict(
        source=frame_rgb,
        conf=0.35,
        iou=0.5,
        device=device,
        show=False
    )
    
    # Color palette
    colors = np.random.randint(0, 255, (1000, 3), dtype=np.uint8)
    
    # Process results
    for result in results:
        processed_frame = result.orig_img.copy()
        sam_predictor.set_image(frame_rgb)
        
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        
        for idx, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
            x1, y1, x2, y2 = map(int, box)
            
            # Generate smart points - center + two diagonal points
            center = [int((x1 + x2) / 2), int((y1 + y2) / 2)]
            quarter_w = (x2 - x1) // 4
            quarter_h = (y2 - y1) // 4
            fg_points = [
                center,
                [x1 + quarter_w, y1 + quarter_h],  # top-left quadrant
                [x2 - quarter_w, y2 - quarter_h]   # bottom-right quadrant
            ]
            
            # Add one background point outside the box
            bg_point = [max(0, x1 - 10), center[1]]
            
            input_points = np.array(fg_points + [bg_point])
            input_labels = np.array([1, 1, 1, 0])  # Last point is background
            
            # Bounding box prompt
            sam_box = np.array([x1, y1, x2, y2])
            
            # Get mask from SAM
            masks, scores, _ = sam_predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                box=sam_box,
                multimask_output=False
            )
            
            # Process mask with higher threshold
            mask = masks[0].cpu().numpy() if isinstance(masks[0], torch.Tensor) else masks[0]
            mask = (mask > 0.65).astype(np.uint8)
            mask = clean_mask(mask)
            
            # Find mask boundaries
            y_indices, x_indices = np.where(mask == 1)
            if len(y_indices) == 0:
                continue
                
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            width = x_max - x_min
            height = y_max - y_min
            side = max(width, height)
            center_x_mask = (x_min + x_max) // 2
            center_y_mask = (y_min + y_max) // 2

            # Calculate square coordinates
            x1_sq = center_x_mask - side // 2
            y1_sq = center_y_mask - side // 2
            x2_sq = x1_sq + side
            y2_sq = y1_sq + side

            # Calculate padding needed
            pad_left = max(0, -x1_sq)
            pad_right = max(0, x2_sq - frame_rgb.shape[1])
            pad_top = max(0, -y1_sq)
            pad_bottom = max(0, y2_sq - frame_rgb.shape[0])

            # Adjust coordinates to fit within the image
            x1_adj = max(0, x1_sq)
            y1_adj = max(0, y1_sq)
            x2_adj = min(frame_rgb.shape[1], x2_sq)
            y2_adj = min(frame_rgb.shape[0], y2_sq)

            # Crop original image and mask
            cropped_original = frame_rgb[y1_adj:y2_adj, x1_adj:x2_adj]
            cropped_mask = mask[y1_adj:y2_adj, x1_adj:x2_adj]

            # Pad the cropped regions
            padded_original = cv2.copyMakeBorder(cropped_original, pad_top, pad_bottom, pad_left, pad_right,
                                               cv2.BORDER_CONSTANT, value=(0, 0, 0))
            padded_mask = cv2.copyMakeBorder(cropped_mask, pad_top, pad_bottom, pad_left, pad_right,
                                           cv2.BORDER_CONSTANT, value=0)

            # Apply strict masking using bitwise operation
            masked_card = cv2.bitwise_and(padded_original, padded_original, mask=padded_mask)

            # Resize to standard square size (640x640)
            resized_square = cv2.resize(masked_card, (640, 640), interpolation=cv2.INTER_NEAREST)

            # Save the normalized square image
            card_output_path = os.path.join('images', f'card_{idx}_{cls}_{conf:.2f}.jpg')
            cv2.imwrite(card_output_path, cv2.cvtColor(resized_square, cv2.COLOR_RGB2BGR))
            
            # Apply color overlay for visualization
            color = [255, 0, 0]
            overlay = processed_frame.copy()
            overlay[mask == 1] = color
            cv2.addWeighted(overlay, 0.5, processed_frame, 0.5, 0, processed_frame)
            
            # Draw bounding box and label
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{result.names[int(cls)]} {conf:.2f}"
            cv2.putText(processed_frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save and display results
        cv2.imwrite(output_path, cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
        cv2.imshow('Detection & Segmentation', cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # Example usage
    input_image = r'mtg_cards_for_final_project.jpg'
    process_image(input_image, 'images/output_image.jpg')