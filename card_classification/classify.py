import os
import sys
import pickle
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torch.nn.functional import normalize
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from card_classification.model.siamese_resnet import SiameseNetwork
from utils.cardDatasetUtils import CardImageDataset

# ------------------- Config -------------------
EMBEDDING_PATH = "embeddings/card_embeddings.pkl"
MODEL_PATH = "model/siamese_model.pth"
IMAGE_SIZE = 128

# ------------------- Setup -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# ------------------- Load Model & Embeddings -------------------
def load_model(path=MODEL_PATH):
    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def load_embeddings(path=EMBEDDING_PATH):
    with open(path, "rb") as f:
        data = pickle.load(f)
    embeddings = normalize(data["embeddings"], dim=1)
    labels = np.array(data["labels"])
    return embeddings, labels

# Load card info mapping (label -> id)
def load_card_info(csv_path):
    card_info_df = pd.read_csv(csv_path)
    label_to_id = dict(zip(card_info_df['mantle_sku'], zip(card_info_df['id'], card_info_df['product_name'])))
    return label_to_id


# ------------------- Inference Helpers -------------------
def classify_top_k_fast(model, image_tensor, all_embeddings, all_labels, k=5):
    """Returns top-k most likely labels and their distances for a given image tensor."""
    with torch.no_grad():
        embedding = model.forward_one(image_tensor.unsqueeze(0).to(device))
        embedding = normalize(embedding, dim=1).cpu()
        
        # Compute distances to all reference embeddings
        distances = torch.norm(all_embeddings - embedding, dim=1)
        
        # Find top-k closest
        topk = torch.topk(distances, k, largest=False)
        topk_indices = topk.indices
        topk_distances = topk.values

        # Return both labels and distances
        topk_labels = [all_labels[i] for i in topk_indices]

        return topk_labels



def classify_top_k(model, image_tensor, all_embeddings, all_labels, label_to_id, k=5):
    """Returns top-k most likely labels and their distances for a given image tensor."""
    with torch.no_grad():
        image_tensor = transform(image_tensor)
        embedding = model.forward_one(image_tensor.unsqueeze(0).to(device))
        
        print(f"Norms before normalization: {torch.norm(embedding)}")
        embedding = normalize(embedding, dim=1).cpu()
        print(f"Norms after normalization: {torch.norm(embedding)}")
        
        print(f"Norms before normalization (individual): {torch.norm(all_embeddings, dim=1)}")
        # Normalize the embeddings
        all_embeddings = normalize(all_embeddings, dim=1).cpu()
        print(f"Norms after normalization (individual): {torch.norm(all_embeddings, dim=1)}")
        
        # Compute distances to all reference embeddings
        distances = torch.norm(all_embeddings - embedding, dim=1)
        
        # Find top-k closest
        topk = torch.topk(distances, k, largest=False)
        topk_indices = topk.indices
        topk_distances = topk.values

        # Return both labels and distances
        topk_labels = [all_labels[i] for i in topk_indices]
        topk_distances = [round(d.item(), 10) for d in topk_distances]
        topk_names = []

        # Plot top-k matches
        for i in range(k):
            topk_names.append(label_to_id[topk_labels[i]][1])

        return list(zip(topk_labels, topk_distances, topk_names))

# Function to classify top-k and plot images
def classify_top_k_and_plot(model, image_tensor, all_embeddings, all_labels, image_dir, label_to_id, k=3):
    """Returns top-k most likely labels and their distances, and plots the test image with top-k matches."""
    with torch.no_grad():
        image_tensor = transform(image_tensor)  # Transform input image tensor
        embedding = model.forward_one(image_tensor.unsqueeze(0).to(device))
        embedding = normalize(embedding, dim=1).cpu()

        distances = torch.norm(all_embeddings - embedding, dim=1)
        topk = torch.topk(distances, k, largest=False)
        topk_indices = topk.indices
        topk_distances = topk.values

        topk_labels = [all_labels[i] for i in topk_indices]
        topk_distances = [round(d.item(), 3) for d in topk_distances]

        # Plot the test image
        fig, axes = plt.subplots(1, k+1, figsize=(15, 5))  # +1 for the test image itself
        axes[0].imshow(image_tensor.permute(1, 2, 0).cpu())  # Convert tensor to HWC format
        axes[0].set_title("Test Image")
        axes[0].axis("off")

        # Plot top-k matches
        for i in range(k):
            image_id = label_to_id[topk_labels[i]][0]  # Get the ID for the label
            
            # Try to load the image as .jpg or .png
            img_path_jpg = f"{image_dir}/{image_id}.jpg"
            img_path_png = f"{image_dir}/{image_id}.png"
            
            # Check which file exists and load it
            if os.path.exists(img_path_jpg):
                img_path = img_path_jpg
            elif os.path.exists(img_path_png):
                img_path = img_path_png
            else:
                print(f"Error: Image {image_id} not found as .jpg or .png.")
                continue

            img = mpimg.imread(img_path)
            axes[i+1].imshow(img)
            axes[i+1].set_title(f"{topk_labels[i]} ({topk_distances[i]})")
            axes[i+1].axis("off")

        plt.tight_layout()
        plt.show()

        # Return top-k labels and distances for reference
        return list(zip(topk_labels, topk_distances))



def evaluate_model_accuracy(model, dataset, all_embeddings, all_labels, k=1, num_samples=None):
    """Evaluates top-k accuracy on given dataset."""
    correct, total = 0, 0

    loader = list(dataset)
    if num_samples is not None:
        loader = loader[:num_samples]

    for img, true_label in tqdm(loader, desc=f"Evaluating Top-{k}"):
        predicted_labels = classify_top_k_fast(model, img.to(device), all_embeddings, all_labels, k=k)
        if true_label in predicted_labels:
            correct += 1
        total += 1

    acc = correct / total
    print(f"\nTop-{k} Accuracy: {acc:.2%} ({correct}/{total})")
    return acc

# ------------------- Example Usage -------------------
if __name__ == "__main__":
    from utils.cardDatasetUtils import CardImageDataset

    TEST_CSV_PATH = "../data_generator/test_info.csv"
    TEST_IMG_DIR = "../test_images"
    TOP_K = 3
    NUM_SAMPLES = None  # Set to None to use full test set

    # Load everything
    model = load_model()
    all_embeddings, all_labels = load_embeddings()
    test_dataset = CardImageDataset(csv_path=TEST_CSV_PATH, image_dir=TEST_IMG_DIR, transform=transform)

    # Run evaluation
    evaluate_model_accuracy(model, test_dataset, all_embeddings, all_labels, k=TOP_K, num_samples=NUM_SAMPLES)

    # Classify a single image (example)
    # image = Image.open("path/to/card.jpg").convert("RGB")
    # image_tensor = transform(image)
    # top_labels = classify_top_k(model, image_tensor, all_embeddings, all_labels, k=TOP_K)
    # print("Top-k predictions:", top_labels)
