import cv2
import torch
import numpy as np
from ultralytics import YOLO
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def run_inference():
    # Load YOLO model
    model = YOLO('runs/detect/train2/weights/best.pt')  # Update path
    
    # Initialize SAM
    sam_checkpoint = "sam2-main/checkpoints/sam2.1_hiera_base_plus.pt"  # Update path
    sam_config = r"C:\Users\willi\Documents\GitHub\DeckTection\Card-Detection\sam2-main\sam2\configs\sam2.1\sam2.1_hiera_b+.yaml"    # Update path (only works with absolute path idk why)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam_model = build_sam2(sam_config, sam_checkpoint)
    sam_model.to(device)
    sam_predictor = SAM2ImagePredictor(sam_model)
    
    # Video source
    source_path = r'C:\Users\willi\Documents\GitHub\DeckTection\Card-Detection\Streamer pulls Rainbow Pikachu on a pack he got paid to bend. $200 card.mp4'
    
    # Get video properties
    cap = cv2.VideoCapture(source_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('detection_segmentation_output.mp4', fourcc, fps, (width, height))
    
    # Color palette for masks
    colors = np.random.randint(0, 255, (1000, 3), dtype=np.uint8)
    
    # Run inference
    results = model.predict(
        source=source_path,
        conf=0.35,
        iou=0.5,
        device=device,
        show=False,  # Disable YOLO's built-in display
        stream=True
    )
    
    # Process results
    for result in results:
        frame = result.orig_img.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        sam_predictor.set_image(frame_rgb)
        
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        
        # Process each detection
        for box, cls, conf in zip(boxes, classes, confidences):
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
            
            # Remove the 'if masks:' check and handle directly
            mask = masks[0]  # We get first mask since multimask_output=False
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            mask = (mask > 0.5).astype(np.uint8)  # Threshold mask
            
            # Apply color overlay
            color = colors[int(cls) % 1000].tolist()
            overlay = frame.copy()
            overlay[mask == 1] = color
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
            # Draw bounding box and label (keep this after segmentation)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{result.names[int(cls)]} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Write frame to output video
        out.write(frame)
        
        # Optional: Display frame
        cv2.imshow('Detection & Segmentation', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    
    # Cleanup
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_inference()