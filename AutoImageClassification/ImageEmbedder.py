import torch
from torchvision import transforms
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights
import torch.nn as nn
# 
import numpy as np
import pandas as pd
from PIL import Image
import concurrent.futures
import os
import zipfile

#----------------------Creating Embedding features for images using pretrained EfficientNetB7 model-----------------
# 
class ImageEmbedder:
    def __init__(self, batch_size=16, device=None, num_augmentation = 5):
        """
        Args:Images_path (str): Root directory containing subfolders of categorized images.
            Output_path (str): Directory where the embeddings will be saved.
            batch_size (int): Number of images to process per batch during embedding.
            device (str or None): Computation device ('cuda', 'cpu', or None to auto-select).
            num_augmentation (int): Number of augmented versions per image if augmentation is enabled.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu") #  If supported, using GPU for image embedding
        self.batch_size = batch_size
        self.num_augmentation = num_augmentation

        # Load EfficientNetB7 and remove the classifier head
        weights = EfficientNet_B7_Weights.DEFAULT
        model = efficientnet_b7(weights=weights)
        self.embedding_model = nn.Sequential(
            model.features,
            model.avgpool,
            nn.Flatten(),
        ).to(self.device)
        self.embedding_model.eval()

        self.preprocess = transforms.Compose([
            transforms.Resize(600),
            transforms.CenterCrop(600),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]), # Normalize using ImageNet stats (used during B7 pretraining)
        ])
        
        self.preprocess_aug = transforms.Compose([
            transforms.Resize(600),
            transforms.CenterCrop(600),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]), 
        ])

    # Process image and return it as a list of tensors (containing only one tensor)
    def _preprocess_image(self, path):
        img = Image.open(path).convert("RGB")
        augmented_tensors = []
        processed_img = self.preprocess(img).unsqueeze(0) # Add batch dim
        augmented_tensors.append(processed_img) # To have a same return structure as _preprocess_image_augment function
        return augmented_tensors  # List of tensors

    # Process image and perform image augmentation
    def _preprocess_image_augment(self, path):
        img = Image.open(path).convert("RGB")
        augmented_tensors = []
        for _ in range(self.num_augmentation):
            aug_img = self.preprocess_aug(img).unsqueeze(0)  
            augmented_tensors.append(aug_img)
        return augmented_tensors  # List of tensors
    
    # Generates image embeddings from a list of image file paths and returns a list of NumPy arrays representing image embeddings.
    # Uses multithreading to speed up image preprocessing
    def embed_images(self, image_paths, augmentation = False):

        all_embeddings = []
        all_image_paths = []
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for i in range(0, len(image_paths), self.batch_size):
                batch_paths = image_paths[i:i + self.batch_size]
                batch_augmented_lists = []
                if augmentation:
                     # Parallel image loading and preprocessing
                    batch_augmented_lists = list(executor.map(self._preprocess_image_augment, batch_paths)) # list of list of tensors: shape (5,batch_size)
                else:
                    batch_augmented_lists = list(executor.map(self._preprocess_image, batch_paths))
                # Creating a list containing processed tensors of all the images in batches and all the augmented images
                batch_tensors = [t for sublist in batch_augmented_lists for t in sublist]
                batch = torch.cat(batch_tensors).to(self.device)
    
                with torch.no_grad():
                    embeddings = self.embedding_model(batch).cpu().numpy()
                    all_embeddings.extend(embeddings)
    
                print('Feed Forward Finished. Batch:', i)
    
        return all_embeddings


    # Create embedding features, and image augmentations for all the images stored in subfolders of Images_path directory and store them into Output_path
    # Creating one .npz for each category
    def Embedding(self, Images_path, Output_path, augment = False):
        os.makedirs(Output_path, exist_ok=True)
        
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
        # Subdirectories in Images_path cotaining images
        directories = [d for d in os.listdir(Images_path) if os.path.isdir(os.path.join(Images_path, d))]
        Categories = []
        
        for idx, directory in enumerate(directories, start=0):
            embedding_file = os.path.join(Output_path, f"embeddings_{idx}.npz")
            if os.path.exists(embedding_file):
                continue
        
            print(f"\nProcessing Category {idx}: {directory}")
        
            dir_path = os.path.join(Images_path, directory)
            image_files = [
                os.path.join(dir_path, f)
                for f in os.listdir(dir_path)
                if os.path.isfile(os.path.join(dir_path, f)) and os.path.splitext(f)[1].lower() in image_extensions
            ]
        
            if not image_files:
                print(f"  No images found in {directory}")
                continue
        
            embeddings = self.embed_images(image_files, augmentation = augment)

            # Repeat each path num_augmentation times to match augmented embeddings
            if augment:
                all_image_paths = []
                for image_file in image_files:
                    all_image_paths.extend([image_file] * self.num_augmentation)
                image_files = all_image_paths

            # for each image store Image_Path, Category_Index, and an array containg Embedding features
            records = [
                {"Image_Path": path, "Category_Index": idx, "Embedding": emb}
                for path, emb in zip(image_files, embeddings)
            ]
        
            # Save per-category .npz file
            np.savez(embedding_file, embeddings=records)
            del records
            print(f"  Saved embeddings for {len(image_files)} images.")
        
        # Save categories info into the root folder
        df_Categories = pd.DataFrame([{"index": i, "Category": d} for i, d in enumerate(directories, start=0)])
        df_Categories.to_csv("Categories.csv", index=False)
        print("\n✅ All categories processed.")

    # Return the embedding features for one image 
    def embed_one_image(self, image_path):
        """Takes a single image path and returns its numpy embedding""" 
        # Preprocess the image
        img = Image.open(image_path).convert("RGB")
        tensor = self.preprocess(img).unsqueeze(0) 
        image_tensor = tensor.to(self.device)
    
        with torch.no_grad():
            embedding = self.embedding_model(image_tensor).cpu().numpy()
        return embedding


    # Function to load all the saved embedding feature files (.npz) in a folder
    # combine them and return them in one dataframe 
    @staticmethod
    def Loading_Embeddings(folder_path):
        all_images = []
        for filename in os.listdir(folder_path):
            if filename.endswith('.npz'):
                file_path = os.path.join(folder_path, filename)
                
                if not zipfile.is_zipfile(file_path):
                    print(f"Skipping invalid zip file: {filename}")
                    continue
                # Load .npz file
                data = np.load(file_path, allow_pickle=True)
                # Extract arrays
                embeddings = data["embeddings"]  # shape: (N, 2560)
                # Convert to DataFrame
                for item in embeddings:
                    Image_Path = item['Image_Path']
                    idx = item['Category_Index']
                    emb = item['Embedding'].flatten()  # convert (1, 2560) to (2560,)
                    all_images.append([Image_Path,idx,emb])  # prepend index to the row
        # Create DataFrame
        df = pd.DataFrame(all_images, columns=['Image_Path','Cat_Index','Embedding'])
        return df

    # Load one embedding file
    @staticmethod
    def Load_one_embedding(file_path):
        # Load .npz file
        data = np.load(file_path, allow_pickle=True)
        # Extract arrays
        embeddings = data["embeddings"]  # shape: (N, 2560)
        # Convert to DataFrame
        records = []
        for item in embeddings:
            Image_Path = item[0]
            idx = item[1]
            emb = item[2].flatten()  # convert (1, 2560) to (2560,)
            records.append([Image_Path,idx,emb])  # prepend index to the row
        # Create DataFrame
        df = pd.DataFrame(records, columns=['Image_Path','Cat_Index','Embedding'])
        return df

