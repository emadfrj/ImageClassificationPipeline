#!/usr/bin/env python
# coding: utf-8

# In[11]:


# Importing neccessary moduls
# AutoImageClassification library moduls
import AutoImageClassification.Bing_Image_Download as BID 
import AutoImageClassification.ImageEmbedder as IE 
import AutoImageClassification.Anomaly as AN 
import AutoImageClassification.ImageClassification as IC

import os
import pandas as pd
import numpy as np

class image_classification_prediction:
    def __init__(self, model_path, Categories_path):
        # Embeding model
        self.embedder = IE.ImageEmbedder()
        # Load classification model
        Model_path = os.path.join(model_path)
        self.Model_loaded = IC.ImageClassification()
        self.Model_loaded.load_model(Model_path)
        # Category names
        self.Categories = pd.read_csv(Categories_path)

    def prediction(self,image_path):
        embedded_features = self.embedder.embed_one_image(image_path) 
        df = pd.DataFrame([{"Embedding": embedded_features}])
        index = self.Model_loaded.predict_class(df['Embedding'][0])
        Cat = self.Categories
        Category_name = Cat.loc[Cat['index'] == index[0], 'Category'].values[0]
        return Category_name

