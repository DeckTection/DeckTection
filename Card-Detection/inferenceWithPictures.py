import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def process_image(image_path, output_path='images/output.jpg'):
    # Create output directories if they don't exist
    os.makedirs('images', exist_ok=True)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Load YOLO model
    model = YOLO('runs/detect/train2/weights/best.pt')  # Update path
    
    # Initialize SAM
    sam_checkpoint = "sam2-main/checkpoints/sam2.1_hiera_base_plus.pt"  # Update path
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
    
    # Convert to RGB for processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Run YOLO detection
    results = model.predict(
        source=frame_rgb,
        conf=0.35,
        iou=0.5,
        device=device,
        show=False
    )
    
    # Color palette for masks
    colors = np.random.randint(0, 255, (1000, 3), dtype=np.uint8)
    
    # Process results
    for result in results:
        # Make copy of original image
        processed_frame = result.orig_img.copy()
        sam_predictor.set_image(frame_rgb)
        
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        
        # Process each detection
        for idx, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
            x1, y1, x2, y2 = map(int, box)
            
            # Generate center point prompt for SAM
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            input_point = np.array([[center_x, center_y]])
            input_label = np.array([1])  # 1 indicates foreground
            
            # Get segmentation mask from SAM
            masks, scores, _ = sam_predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False  # Single mask output
            )
            
            # Process mask
            mask = masks[0]
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            mask = (mask > 0.5).astype(np.uint8)
            
            # Normalize mask to square
            y_indices, x_indices = np.where(mask == 1)
            if len(y_indices) == 0 or len(x_indices) == 0:
                continue  # Skip if no mask found
            
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

            # Apply mask to isolate the card
            padded_original[padded_mask == 0] = 0

            # Resize to standard square size (e.g., 640x640)
            resized_square = cv2.resize(padded_original, (640, 640))

            # Save the normalized square image
            card_output_path = os.path.join('images', f'card_{idx}_{cls}_{conf:.2f}.jpg')
            cv2.imwrite(card_output_path, cv2.cvtColor(resized_square, cv2.COLOR_RGB2BGR))
            
            # Apply color overlay for visualization
            color = colors[int(cls) % 1000].tolist()
            overlay = processed_frame.copy()
            overlay[mask == 1] = color
            cv2.addWeighted(overlay, 0.3, processed_frame, 0.7, 0, processed_frame)
            
            # Draw bounding box and label
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{result.names[int(cls)]} {conf:.2f}"
            cv2.putText(processed_frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save and display results
        cv2.imwrite(output_path, cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
        cv2.imshow('Detection & Segmentation', cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # Example usage
    input_image = r'mtg_cards_for_final_project.jpg'  # Update with your image path
    process_image(input_image, 'images/output_image.jpg')