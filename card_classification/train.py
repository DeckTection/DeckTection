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

siamese_dataset = SiameseDataset(dataset)

# Use DataLoader to load data in batches
dataloader = DataLoader(siamese_dataset, batch_size=32, shuffle=True)

# Instantiate model and optimizer
model = SiameseNetwork().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = ContrastiveLoss()


num_epochs = 100
import math
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    
    for idx, (img1, img2, label) in enumerate(dataloader):
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        output1, output2 = model(img1, img2)
        
        # Compute contrastive loss
        loss = loss_fn(output1, output2, label)
        
        # Backpropagate and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if idx % 100 == 0:
            with torch.no_grad():
                # Check average distance for same/diff class
                dists = torch.norm(output1 - output2, dim=1)
                same = label == 0
                diff = label == 1
                print(f"[Epoch {epoch} | Batch {idx}] Loss: {loss.item():.4f}")
                print(f"  Avg same-class dist: {dists[same].mean().item():.4f}")
                print(f"  Avg diff-class dist: {dists[diff].mean().item():.4f}")

    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}')



torch.save(model.state_dict(), "model/siamese_model.pth")



