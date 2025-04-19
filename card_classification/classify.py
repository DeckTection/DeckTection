import os
import sys
import pickle
import torch
from torch.nn.functional import normalize, pairwise_distance
from torchvision import transforms
from tqdm import tqdm
import numpy as np

# Import local project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.siamese_resnet import SiameseNetwork
from utils.cardDatasetUtils import CardImageDataset

# ------------------- Configuration -------------------
EMBEDDING_PATH = "embeddings/cifar10_embeddings.pkl"
MODEL_PATH = "model/siamese_model.pth"
TEST_CSV_PATH = "../data_generator/test_info.csv"
TEST_IMG_DIR = "../test_images"
IMAGE_SIZE = 128
TOP_K = 1  # Change to 3, 5, etc. to evaluate top-k accuracy

# ------------------- Setup -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model = SiameseNetwork()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Load reference embeddings
with open(EMBEDDING_PATH, "rb") as f:
    data = pickle.load(f)
    all_embeddings = normalize(data["embeddings"], dim=1)
    all_labels = data["labels"]

# ------------------- Dataset -------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

test_dataset = CardImageDataset(csv_path=TEST_CSV_PATH, image_dir=TEST_IMG_DIR, transform=transform)

# ------------------- Inference -------------------
def find_nearest_neighbors(query_img_tensor, k=1):
    with torch.no_grad():
        query_embedding = model.forward_one(query_img_tensor.unsqueeze(0).to(device))
        query_embedding = normalize(query_embedding, dim=1).cpu()

        distances = torch.norm(all_embeddings - query_embedding, dim=1)  # Euclidean
        topk = torch.topk(distances, k, largest=False)
        return topk.indices, all_labels[topk.indices]

def evaluate_model(dataset, k=1):
    correct = 0
    total = 0

    for img, true_label in tqdm(dataset, desc="Evaluating"):
        indices, predicted_labels = find_nearest_neighbors(img.to(device), k=k)
        # predicted_labels is likely a NumPy array or list of strings
        if isinstance(predicted_labels, (list, np.ndarray)):
            top_k_labels = list(predicted_labels)
        else:
            top_k_labels = [predicted_labels]  # force into list

        if true_label in top_k_labels:
            correct += 1
        total += 1

    acc = correct / total
    print(f"\nTop-{k} Accuracy: {acc:.2%} ({correct}/{total})")

# ------------------- Run -------------------
if __name__ == "__main__":
    evaluate_model(test_dataset, k=TOP_K)
