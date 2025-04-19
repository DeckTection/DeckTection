import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.models as models
from torchvision import datasets
from PIL import Image
import numpy as np
from model.siamese_resnet import SiameseDataset, SiameseNetwork, ContrastiveLoss
import pickle
import sys
import os
# Add the root of your project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now you can do a normal import
from utils.cardDatasetUtils import load_card_dataset_pkl, CardImageDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example of applying the CardImageDataset and transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to 32x32 like CIFAR-10
    transforms.ToTensor(),        # Convert the image to a tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize like CIFAR-10
])

dataset = CardImageDataset(csv_path="../data_generator/card_info.csv", image_dir="../card_images", transform=transform)

# Instantiate model and optimizer
model = SiameseNetwork().to(device)
model.load_state_dict(torch.load("model/siamese_model.pth"))


# Assumes your model is already trained and on the right device
model.eval()


embedding_list = []
label_list = []

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


with torch.no_grad():
    for img, label in DataLoader(dataset, batch_size=32):  # use the same transform
        img = img.to(device)
        embeddings = model.forward_one(img)  # assuming this returns the 1D feature vector
        embedding_list.append(embeddings.cpu())
        label_list.append(label)

# Stack into full tensors
all_embeddings = torch.cat(embedding_list)
# Flatten your nested tuple list
flat_labels = [label for group in label_list for label in group]
# Encode to integers
encoder = LabelEncoder()
encoded = encoder.fit_transform(flat_labels)
# Convert to tensor
all_labels = torch.tensor(encoded)

import pickle 

with open("embeddings/cifar10_embeddings.pkl", "wb") as f:
    pickle.dump({
        "embeddings": all_embeddings,
        "labels": flat_labels  # labels
    }, f)
