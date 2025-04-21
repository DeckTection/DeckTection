import os
import sys
import cv2
import torch
from torchvision import transforms
from card_classification.classify import classify_top_k, load_embeddings, classify_top_k_and_plot, load_card_info
from card_classification.embeddings import load_model, generate_initial_embeddings, add_embedding
from card_preprocessing.preprocess import preprocess  # Assumes this handles resize + normalization
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ------------------- Config -------------------
IMAGE_SIZE = 128
TOP_K = 10  # Change as needed
MODEL_PATH = "card_classification/model/siamese_model.pth"
IMAGES_PATH = "card_images"
IMAGE_DATA_PATH = "data_generator/card_info.csv"
EMBEDDING_PATH = "card_classification/embeddings/card_embeddings.pkl"
TEST_CSV = "data_generator/test_info.csv"

# ------------------- Image Loader -------------------
def load_image(image_path, csv_path=TEST_CSV):

    # Load the card info CSV into a DataFrame
    card_info_df = pd.read_csv(csv_path)

    # Extract the filename from the image path (assuming the image filename is the same as the 'id' in the CSV)
    image_filename = image_path.split("/")[-1]  # Extract filename from path
    card_id = image_filename  # Get the card id (remove extension)

    # Look up the row in the DataFrame where 'image_name' matches the card ID
    card_info = card_info_df[card_info_df['image_name'] == card_id]

    if not card_info.empty:
        # Extract values if the row was found
        mantle_sku = card_info.iloc[0]['mantle_sku']
        product_name = card_info.iloc[0]['product_name']
    else:
        # Fallback if no matching row was found
        mantle_sku = "unknown"
        product_name = "unknown"

    # Now load the image itself
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Convert the image to RGB (OpenCV loads images as BGR)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Return both the image and the additional card info
    return rgb_image, mantle_sku, product_name

# ------------------- Main Pipeline -------------------
def main(image_path):

    print("🔍 Loading image...")
    try:
        image, mantle_sku, product_name = load_image(image_path)
        print(f"Loaded image with Mantle SKU: {mantle_sku}, Product Name: {product_name}")
    except Exception as e:
        print(str(e))

    print("🧼 Preprocessing...")
    image_tensor = preprocess(image)  # Should return torch.Tensor of shape [3, H, W]

    print("🧠 Loading model and embeddings...")
    model = load_model(model_path=MODEL_PATH)
    # Check if the embeddings file already exists
    if not os.path.exists(EMBEDDING_PATH):
        print(f"Embeddings not found. Generating initial embeddings...")
        generate_initial_embeddings(
            dataset_path=IMAGE_DATA_PATH, 
            image_dir=IMAGES_PATH,
            model_path=MODEL_PATH,
            output_path=EMBEDDING_PATH
        )
    else:
        print(f"Embeddings already exist at {EMBEDDING_PATH}. Skipping generation.")
        
    all_embeddings, all_labels = load_embeddings(path=EMBEDDING_PATH)

    print(f"📊 Running classification (Top-{TOP_K})...")
    image_data = load_card_info(IMAGE_DATA_PATH)
    top_k_labels = classify_top_k(model, image_tensor, all_embeddings, all_labels, label_to_id=image_data, k=TOP_K)
    # top_k_labels = classify_top_k_and_plot(model, image_tensor, all_embeddings, all_labels,image_dir=IMAGES_PATH,label_to_id=image_data, k=TOP_K)

    print(f"✅ Top-{TOP_K} Predictions:")
    for rank, [label, distance, name] in enumerate(top_k_labels, 1):
        print(f"{rank}. {name} : {label} : {distance}")

# ------------------- CLI Entry -------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main_pipeline.py path/to/image.jpg")
        sys.exit(1)
    main(sys.argv[1])
