import os
import sys
import cv2
import torch
from torchvision import transforms
from card_classification.classify import classify_top_k, load_embeddings, classify_top_k_and_plot, load_card_info
from card_classification.embeddings import load_model, generate_initial_embeddings, add_embedding
from card_preprocessing.preprocess import preprocess  # Assumes this handles resize + normalization

# ------------------- Config -------------------
IMAGE_SIZE = 128
TOP_K = 3  # Change as needed
MODEL_PATH = "card_classification/model/siamese_model.pth"
IMAGES_PATH = "card_images"
IMAGE_DATA_PATH = "data_generator/card_info.csv"
EMBEDDING_PATH = "card_classification/embeddings/card_embeddings.pkl"

# ------------------- Image Loader -------------------
def load_image(image_path):
    """Loads an image from disk and converts to RGB numpy array."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ------------------- Main Pipeline -------------------
def main(image_path):
    print("🔍 Loading image...")
    image = load_image(image_path)

    print("🧼 Preprocessing...")
    image_tensor = preprocess(image)  # Should return torch.Tensor of shape [3, H, W]

    print("🧠 Loading model and embeddings...")
    model = load_model(model_path=MODEL_PATH)
    generate_initial_embeddings(
        dataset_path=IMAGE_DATA_PATH, 
        image_dir=IMAGES_PATH,
        model_path=MODEL_PATH,
        output_path=EMBEDDING_PATH
        )
    
    all_embeddings, all_labels = load_embeddings(path=EMBEDDING_PATH)

    print(f"📊 Running classification (Top-{TOP_K})...")
    top_k_labels = classify_top_k(model, image_tensor, all_embeddings, all_labels, k=TOP_K)
    image_data = load_card_info(IMAGE_DATA_PATH)
    top_k_labels = classify_top_k_and_plot(model, image_tensor, all_embeddings, all_labels,image_dir=IMAGES_PATH,label_to_id=image_data, k=TOP_K)

    print(f"✅ Top-{TOP_K} Predictions:")
    for rank, [label, distance] in enumerate(top_k_labels, 1):
        print(f"{rank}. {label} : {distance}")

# ------------------- CLI Entry -------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main_pipeline.py path/to/image.jpg")
        sys.exit(1)
    main(sys.argv[1])
