import pickle
import torch
from model.siamese_resnet import SiameseNetwork
import torchvision.transforms as transforms
from torchvision import datasets


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = SiameseNetwork()
model.load_state_dict(torch.load('model/siamese_model.pth', map_location=device))
model.to(device)
model.eval()

with open("embeddings/cifar10_embeddings.pkl", "rb") as f:
    data = pickle.load(f)
    all_embeddings = data["embeddings"]
    all_labels = data["labels"]


from torch.nn.functional import pairwise_distance

def find_nearest_neighbors(query_img_tensor, k=1):
    query_embedding = model.forward_one(query_img_tensor.unsqueeze(0).to(device)).cpu()

    # Compute distances to all enrolled samples
    distances = torch.norm(all_embeddings - query_embedding, dim=1)  # Euclidean

    # Find top-k closest embeddings
    topk = torch.topk(distances, k, largest=False)
    return topk.indices, distances[topk.indices], all_labels[topk.indices]

# Define transformations
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Get a test images from CIFAR
for test_img, test_label in test_dataset:
    test_img_tensor = test_img.to(device)
    indices, dists, labels = find_nearest_neighbors(test_img_tensor)
    if test_label == labels[0].item():
        correct += 1
    total +=1

print(f"\nAccuracy: {correct / total:.2%}")