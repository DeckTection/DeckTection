import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from PIL import Image
import numpy as np


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # Load a pre-trained ResNet18 model, excluding the final fully connected layers
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()  # Remove fully connected layer to get embeddings
        
        # Optional: Add an additional fully connected layer for embedding
        self.fc = nn.Sequential(
            nn.Linear(512, 256),  # ResNet18 outputs 512 features
            nn.ReLU(),
            nn.Linear(256, 128)  # Final embedding size
        )

    def forward_one(self, x):
        x = self.resnet(x)  # Get feature embeddings
        x = self.fc(x)      # Apply additional FC layer
        return x

    def forward(self, input1, input2):
        # Forward both images through the same network
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        
        # Return the embeddings
        return output1, output2


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output1, output2, label):
        euclidean_distance = torch.sqrt(torch.sum((output1 - output2) ** 2, dim=1))
        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) + 
                          (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss
    

class SiameseDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img1_path, img2_path = self.image_paths[index]
        label = self.labels[index]
        
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label
    



