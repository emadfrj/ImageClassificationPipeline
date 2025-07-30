# ImageClassificationPipeline

---
A modular Python library for automated image classification using deep learning embeddings, supporting easy training and prediction workflows across custom datasets.

## 📂 Project Structure
```
ImageClassificationPipeline/
├── AutoImageClassification/
│   ├── ImageEmbedding.py         # Embeds images using EfficientNet
│   ├── ImageClassification.py    # Defines the classifier model and training logic
│   ├── Anomaly.py                # Detect anomalies using autoencoder recustruction error
│   ├── Bing_Image_Download.py    # Scrap images from Bing to enrich the labeled dataset
│   ├── __init__.py
├── Crops_classification.ipynb    # Full workflow for Agricultural Crops Image dataset: scrap images, embed, anomaly, train, evaluate
├── Crops_prediction.py           # Class for loading model and predicting single image (deploy model)
├── Example.ipynb                 # Example usage of the trained model
├── requirements.txt              # Dependencies
└── README.md
```
---

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/YourRepoName.git
cd YourRepoName
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🧠 Model Overview

- **Embedding Model**: EfficientNet-B7 (feature extraction)
- **Classifier**: Fully connected layers trained on top of image embeddings
- **Frameworks**: PyTorch, torchvision, scikit-learn

---

## 📘 How to Use

### 📌 Training

Open `Crops_classification.ipynb` to:
- Embed images from categorized folders
- Train a classifier
- Save the model and category mapping

### 📌 Prediction

Use the `Crops_prediction.py` class to load a trained model and predict on new images.

#### Example (`Example.ipynb`):

```python
from Crops_prediction import image_classification_prediction
import os

# Load model
model_path = os.path.join('Models', 'ImageClassification_Model.pth')
categories_path = 'Categories.csv'
model = image_classification_prediction(model_path, categories_path)

# Predict
image_path = 'banana_plant.jpg'
result = model.prediction(image_path)
print(result)
```

---

## 🧾 Requirements

```
torch
torchvision
numpy
pandas
scikit-learn
Pillow
icrawler
```

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## 📁 Input Folder Format

Your training images should be organized as:

```
Images/
├── Banana/
│   ├── img1.jpg
│   ├── img2.jpg
├── Wheat/
│   ├── img1.jpg
│   ├── img2.jpg
...
```

Each subfolder is treated as one category.

---

## ✅ Outputs

- Trained model: `Models/ImageClassification_Model.pth`
- Category mapping: `Categories.csv`
- Embedding files: `.npz` per category

## 🌾 Agricultural Crops Image Classification

A modular image classification system for identifying different types of agricultural crops using deep learning. The pipeline is built with reusable components for image embedding, training, and individual image prediction.

---


## 📌 License

This project is open-source under the MIT License.

---

## 👤 Author

Created by [Your Name]  
Feel free to fork, contribute, or get in touch!
