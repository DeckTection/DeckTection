import cv2
from ultralytics import YOLO
import torch
import os

def run_inference():
    # Load trained model
    model = YOLO('runs/detect/train/weights/best.pt')  # Update path
    
    # Configure paths
    input_folder = "path/to/test_images"    # Folder containing test images
    output_folder = "path/to/detection_results"  # Folder to save annotated images
    os.makedirs(output_folder, exist_ok=True)  # Create output folder if needed
    
    # Run inference on all images in folder
    results = model.predict(
        source=input_folder,
        conf=0.35,
        iou=0.5,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        save=False,  # We'll save manually to control output location
        show=False,   # Disable display
        stream=False  # Disable streaming for image folder
    )
    
    # Process and save results
    for result in results:
        if result.boxes is not None:
            # Get detection data
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            
            # Draw bounding boxes on original image
            frame = result.orig_img.copy()
            for box, cls, conf in zip(boxes, classes, confidences):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{result.names[int(cls)]} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            
            # Save to output folder
            filename = os.path.basename(result.path)
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, frame)
            print(f"Saved result to: {output_path}")

if __name__ == '__main__':
    run_inference()