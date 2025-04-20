import os
import sys
import torch
import pickle
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
import torchvision.transforms as transforms

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.cardDatasetUtils import load_card_dataset_pkl, CardImageDataset
from card_classification.model.siamese_resnet import SiameseDataset, SiameseNetwork, ContrastiveLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def load_model(model_path="model/siamese_model.pth"):
    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def generate_initial_embeddings(dataset_path, image_dir, model_path="model/siamese_model.pth", output_path="embeddings/cards_embeddings.pkl", batch_size=32):
    dataset = CardImageDataset(csv_path=dataset_path, image_dir=image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    model = load_model(model_path)

    embedding_list = []
    label_list = []
    with torch.no_grad():
        for img, label in dataloader:
            img = img.to(device)
            embeddings = model.forward_one(img)
            embedding_list.append(embeddings.cpu())
            label_list.extend(label)
    all_embeddings = torch.cat(embedding_list)

    with open(output_path, "wb") as f:
        pickle.dump({
            "embeddings": all_embeddings,
            "labels": label_list
        }, f)
    print(f"Saved {len(label_list)} embeddings to {output_path}")


def add_embedding(image_path, label, embeddings_path="embeddings/card_embeddings.pkl"):
    model = load_model()

    with open(embeddings_path, "rb") as f:
        data = pickle.load(f)

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model.forward_one(image).cpu()

    # Append new embedding and label
    data["embeddings"] = torch.cat([data["embeddings"], embedding])
    data["labels"] = np.append(data["labels"], label)

    with open(embeddings_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Added embedding for label '{label}' to {embeddings_path}")


# # Create all embeddings
# generate_initial_embeddings(
#     dataset_path="decktection/data_generator/card_info.csv",
#     image_dir="../card_images"
# )