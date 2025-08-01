# ImageClassificationPipeline

---
A modular Python library for automated image classification. It enriches labeled datasets by scraping and downloading images from Bing. To ensure data quality, it incorporates autoencoder-based anomaly detection and pseudo-labeling to filter out irrelevant images. A pretrained EfficientNet-B7 model (without the classifier head) generates image embeddings, which serve as the input for both anomaly detection and classification models. All core components are located in the ```AutoImageClassification``` folder.

## 📂 Project Structure
```
ImageClassificationPipeline/
├── AutoImageClassification/
│   ├── ImageEmbedding.py         # Performs image augmentation and embeds images using EfficientNet
│   ├── ImageClassification.py    # Defines the classifier model and training logic
│   ├── Anomaly.py                # Detect anomalies using autoencoder reconstruction error
│   ├── Bing_Image_Download.py    # Scrap images from Bing to enrich the labeled dataset
│   ├── __init__.py
├── Crops_classification.ipynb    # Full workflow for Agricultural Crops Image dataset
├── Crops_prediction.py           # Class for loading model and predicting a single image (deploy model)
├── requirements.txt              # Dependencies
└── README.md
```
---

## 📁 Input Folder Format

Your training images(labeled dataset) should be organized as:

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
The downloaded images will also be organized as such.

---


## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/emadfrj/ImageClassificationPipeline.git
cd ImageClassificationPipeline
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🧠 Model Overview

- **Image Scrapper**: BingImageCrawler library
- **Embedding Model**: EfficientNet-B7 (feature extraction)
- **Alomany Model**: Find anomalies using autoencoder and reconstruction error 
- **Classifier**: Fully connected layers trained on top of image embeddings for pseudo-labeling and final classifier
- **Frameworks**: PyTorch, torchvision, scikit-learn

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

---
## 🌾 Agricultural Crops Image Classification
As an example, the AutoImageClassification library is used to create an agricultural crops image classifier. The labeled agricultural crops dataset is downloaded from [Kaggle](https://www.kaggle.com/datasets/mdwaquarazam/agricultural-crops-image-classification).
The Crops_classification.ipynb file is a Jupyter notebook to run the workflow in steps to train and save the classifier model.  
The Crops_prediction.py contains the image_classification_prediction class for loading the trained model and classifying images. It can be used in deployment.

## ✅ Outputs

- Trained model: `Models/ImageClassification_Model.pth`
- Category mapping: `Categories.csv`
- Embedding files: `.npz` per category for both labeled and downloaded images

---
## 📘 How to Use

### 📌 Training
1. **Prepare Your Dataset**  
   Place your labeled dataset folder in the root directory. The structure should follow the format described in the [Input Folder Format](#-input-folder-format) section.
   
2. **Adjust the Notebook**  
   Open `Crops_classification.ipynb` and modify paths or parameters to match your dataset and use case.
3. **Run the Workflow**  
   Execute the cells in `Crops_classification.ipynb` to go through the full image classification pipeline, which includes:
   - Scraping images for each category
   - Embedding and augmenting images
   - Training an anomaly detection model to flag irrelevant scraped images
   - Training a pseudo-labeling model to predict labels for scraped images
   - Filtering the scraped images using both the anomaly detector and pseudo-labeler
   - Training the final classification model using both your labeled dataset and the filtered scraped data
   - Saving the trained model and category mappings
     
### 📌 Prediction

Use the `Crops_prediction.py` class to load a trained model and predict on new images.
Here is the example usage of the image_classification_prediction class:

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

## 👤 Author

Created by Emad Farjami

Feel free to fork, contribute, or get in touch!
