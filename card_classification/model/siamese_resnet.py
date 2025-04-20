import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from PIL import Image
import numpy as np
import random

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
        euclidean_distance = torch.sqrt(torch.sum((output1 - output2) ** 2, dim=1) + 1e-9)
        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) + 
                          (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss
    

class SiameseDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.class_to_idx = self._get_class_to_idx()


    def __len__(self):
        return len(self.dataset)
    
    def _get_class_to_idx(self):
        """Helper method to map classes to indices."""
        class_to_idx = {}
        for idx, (_, label) in enumerate(self.dataset):
            if label not in class_to_idx:
                class_to_idx[label] = []
            class_to_idx[label].append(idx)
        return class_to_idx

    def __getitem__(self, index):

        img1, label1 = self.dataset[index]
        
        # Find a random image that is either of the same class (positive) or different class (negative)
        same_class = random.choice([True, False])

        if same_class:
            # Select another image from the same class
            same_class_idxs = self.class_to_idx.get(label1, [])
            if len(same_class_idxs) > 1:
                idx = random.choice(same_class_idxs)
                img2, label2 = self.dataset[idx]
            else:
                # If only one image of this class, reuse the same image (to avoid index error)
                img2, label2 = img1, label1
        else:
            # Select a random image from a different class
            idx = index
            while self.dataset[idx][1] == label1:
                idx = random.randint(0, len(self.dataset) - 1)
            img2, label2 = self.dataset[idx]

        # Apply transformations if any
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, int(not same_class)




