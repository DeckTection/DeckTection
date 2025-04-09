import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from PIL import Image
import numpy as np
from model.siamese_resnet import SiameseDataset, SiameseNetwork, ContrastiveLoss



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define image transformation (resize, to tensor, normalize)
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Example data (just for illustration purposes)
image_pairs = [("path_to_img1", "path_to_img2"), ("path_to_img3", "path_to_img4")]  # List of image pairs
labels = [1, 0]  # Labels (1 = same class, 0 = different class)

# Create Dataset and DataLoader
dataset = SiameseDataset(image_pairs, labels, transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Instantiate model and optimizer
model = SiameseNetwork().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = ContrastiveLoss()


num_epochs = 10

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    
    for img1, img2, label in dataloader:
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

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}')



# For testing
model.eval()  # Set model to evaluation mode

# Example: Get the embeddings for a new pair
img1 = Image.open("path_to_new_image1").convert('RGB')
img2 = Image.open("path_to_new_image2").convert('RGB')

img1 = transform(img1).unsqueeze(0).to(device)  # Add batch dimension
img2 = transform(img2).unsqueeze(0).to(device)

with torch.no_grad():
    output1, output2 = model(img1, img2)

# Calculate Euclidean distance between the two embeddings
distance = torch.sqrt(torch.sum((output1 - output2) ** 2))
print(f"Euclidean distance: {distance.item()}")