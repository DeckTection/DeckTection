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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load the CIFAR-10 dataset
dataset = datasets.CIFAR10(root='./data', train=True, download=True)

# Take just the first 500 samples
small_dataset = Subset(dataset, range(500))
siamese_dataset = SiameseDataset(dataset, transform=transform)

# Use DataLoader to load data in batches
dataloader = DataLoader(siamese_dataset, batch_size=128, shuffle=True)

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
            emb1 = model(img1)
            emb2 = model(img2)
            with torch.no_grad():
                # Check average distance for same/diff class
                dists = torch.norm(emb1 - emb2, dim=1)
                same = label == 0
                diff = label == 1
                print(f"[Epoch {epoch} | Batch {idx}] Loss: {loss.item():.4f}")
                print(f"  Avg same-class dist: {dists[same].mean().item():.4f}")
                print(f"  Avg diff-class dist: {dists[diff].mean().item():.4f}")

    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}')



# Assumes your model is already trained and on the right device
model.eval()

dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
small_dataset = Subset(dataset, range(500))
embedding_list = []
label_list = []

with torch.no_grad():
    for img, label in DataLoader(dataset, batch_size=128):  # use the same transform
        img = img.to(device)
        embeddings = model.forward_one(img)  # assuming this returns the 1D feature vector
        embedding_list.append(embeddings.cpu())
        label_list.append(label)

# Stack into full tensors
all_embeddings = torch.cat(embedding_list)
all_labels = torch.cat(label_list)


import pickle 

# Save to a file
with open("embeddings/cifar10_embeddings.pkl", "wb") as f:
    pickle.dump({
        "embeddings": all_embeddings,  # torch.Tensor
        "labels": all_labels           # torch.Tensor
    }, f)


torch.save(model.state_dict(), "model/siamese_model.pth")



