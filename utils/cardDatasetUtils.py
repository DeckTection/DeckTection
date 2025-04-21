import os
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import pickle
from sklearn.preprocessing import LabelEncoder

import os
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
import torch

class CardImageDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        """
        Args:
            csv_path (str): Path to the CSV file with image info.
            image_dir (str): Directory where the images are stored.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform or transforms.ToTensor()  # Default to ToTensor if no transform is passed
        self.image_paths = self.data['image_name'].tolist()
        self.product_names = self.data['product_name'].tolist()
        self.labels = self.data['mantle_sku'].tolist()  # Mantle SKU as the label
        
        # You can also store the class mappings here for further use, if needed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load the image
        image_path = os.path.join(self.image_dir, self.image_paths[idx])
        image = Image.open(image_path).convert('RGB')  # Ensure it's in RGB
        
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        
        # Get the corresponding label (Mantle SKU)
        label = self.labels[idx]
        
        return image, label

def load_card_dataset(image_dir="card_images", csv_path="card_info.csv", transform=None):
    return CardImageDataset(csv_path=csv_path, image_dir=image_dir, transform=transform)

def save_dataset(dataset, filepath):
    # Serialize the dataset object using pickle
    with open(filepath, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Dataset saved to {filepath}")

# Load dataset from pickle file
def load_card_dataset_pkl(filepath, transform=None):
    with open(filepath, 'rb') as f:
        dataset = pickle.load(f)
    return dataset