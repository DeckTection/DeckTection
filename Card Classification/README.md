# DeckTection 🎴
**AI-powered trading card detection and recognition using computer vision and one-shot learning.**  
Detect, segment, and identify cards instantly — whether they’re common or ultra-rare.

---

## 🧠 Model Architecture: Siamese Network with ResNet-18

DeckTection leverages a **Siamese Neural Network** built on top of **ResNet-18** to perform **one-shot learning** for card classification.

### 🔍 Why a Siamese Network?

Siamese networks are ideal for tasks with:
- ✅ **Few examples per class** (just one card image is enough!)
- ✅ **Large class spaces** (hundreds or thousands of unique cards)
- ✅ **Visual similarity challenges** (many cards look nearly identical)

### 🧱 Architecture Overview

- **Backbone**: Pretrained `ResNet-18`, modified to output a compact embedding vector for each image.
- **Input**: A pair of card images — one **query** and one **reference**.
- **Output**: A similarity score (0 to 1) measuring how likely the two images represent the same card.
- **Distance Metric**: Cosine or Euclidean distance on embedding vectors.