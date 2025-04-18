# 🃏 DeckTection

**DeckTection** is an intelligent system designed to detect, segment, and classify playing cards in real-world images using a robust multi-stage pipeline. This project brings together powerful computer vision models and a one-shot learning architecture to handle complex card identification scenarios.

---

## 📸 Overview

DeckTection works in three main stages:

1. **Object Detection & Segmentation**  
2. **Image Preprocessing**  
3. **Card Classification**

---

## 🔍 1. Object Detection & Segmentation

We use a combination of:

- **YOLOv12** (You Only Look Once) for fast and accurate object detection of playing cards
- **SAM (Segment Anything Model)** for high-precision segmentation masks

Together, they extract high-quality card crops from complex scenes with multiple overlapping or rotated cards.

---

## 🛠️ 2. Preprocessing Pipeline

Once cards are segmented, we perform preprocessing to normalize the visual representation:

- 📐 **Perspective Correction**  
  Straightens tilted or skewed cards using homography transformations.

- ✨ **CLAHE (Contrast Limited Adaptive Histogram Equalization)**  
  Enhances local contrast in the card images for better feature extraction, especially under poor lighting.

---

## 🧠 3. Classification with Siamese Network

Classification is done using a **Siamese Network** trained with:

- 🔍 **ResNet-18** as the backbone encoder
- ⚖️ **Contrastive Loss** with Euclidean distance metric
- 💡 One-shot learning: only **one example per card class** is needed at inference time!

The model compares a new card to reference embeddings and returns the closest match with high accuracy.

---

## 🧪 Dataset

- Dataset: 📦 CIFAR-10 (used for training/testing the prototype Siamese network)
- Input: Image pairs of cards
- Training: Optimized with **Adam** and tuned for generalization across card types

---

## 🚀 Getting Started

1. Clone the repo:
   ```bash
   git clone https://github.com/your_username/DeckTection.git
   cd DeckTection
