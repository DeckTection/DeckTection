import cv2
from ultralytics import YOLO
import torch

def run_inference():
    # Load trained model
    model = YOLO('runs/detect/train2/weights/best.pt')  # Update path
    
    # Run inference
    results = model.predict(
        source= r'C:\Users\willi\Documents\GitHub\DeckTection\Card-Detection\Streamer pulls Rainbow Pikachu on a pack he got paid to bend. $200 card.mp4',
        conf=0.35,
        iou=0.5,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        show=True,
        stream=True  # For video processing
    )
    
    # Process results
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        
        # Draw bounding boxes
        frame = result.orig_img.copy()
        for box, cls, conf in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{result.names[int(cls)]} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            
        cv2.imshow('Detection', frame)
        if cv2.waitKey(1) == ord('q'):
            break

if __name__ == '__main__':
    run_inference()