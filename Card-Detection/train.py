from ultralytics import YOLO
import torch

def train_yolov12():
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model - use proper YOLOv12 implementation
    model = YOLO('yolov12l.pt').to(device)  # Ensure this model file exists
    
    # Train with augmentation
    results = model.train(
        data='dataset/data.yaml',
        epochs=300,
        batch=-1,  # Automatic batch size
        imgsz=640,
        device=device,
        workers=8,
        optimizer='AdamW',
        lr0=0.001,
        augment=True,
        flipud=0.3,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.2,
        patience=30,
        rect=False  # Disable rectangular training for card detection
        single_cls=True,  # Treat as single-class dataset
        overlap_mask=False
    )

if __name__ == '__main__':
    train_yolov12() 