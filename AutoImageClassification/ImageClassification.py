# Image classification
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import torch.nn as nn
# 
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import itertools
import os

# Prepare Dataset Class
class EmbeddingDataset(Dataset):
    def __init__(self, df):
        self.embeddings = df['Embedding'].tolist()
        self.labels = df['Cat_Index'].tolist()

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        x = torch.tensor(self.embeddings[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

# Define classifier
class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_size, dropout, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

# Image Classification Class
class ImageClassification():
    def __init__(self):
        self.model = None 
        
    # Training and tuning (return the best model and its parameters)
    def Training_Tuning(self, param_grid, df, model_output_path = None,model_output_name ='Best_Model.pth', _test_size=0.2, NUM_EPOCHS = 50):    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        output_number = df['Cat_Index'].nunique()
        input_number = len(df['Embedding'][0])
        
        train_df, val_df = train_test_split(df, test_size = _test_size, stratify=df['Cat_Index'], random_state=42)
        train_dataset = EmbeddingDataset(train_df)
        val_dataset = EmbeddingDataset(val_df)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64)
        
        param_combinations = list(itertools.product(*param_grid.values()))
        best_val_acc = 0.0
        best_model_state = None
        best_params = None
        
        for lr, dropout, hidden_size in param_combinations:
            print(f"Trying lr={lr}, dropout={dropout}, hidden_size={hidden_size}")
        
            model = Classifier(input_number, hidden_size, dropout, output_number).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            loss_fn = nn.CrossEntropyLoss()
        
            for epoch in range(NUM_EPOCHS):
                model.train()
                for x_batch, y_batch in train_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    optimizer.zero_grad()
                    preds = model(x_batch)
                    loss = loss_fn(preds, y_batch)
                    loss.backward()
                    optimizer.step()
        
            # Validation
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    outputs = model(x_val)
                    predicted = torch.argmax(outputs, 1)
                    correct += (predicted == y_val).sum().item()
                    total += y_val.size(0)
        
            acc = correct / total
            print(f"Validation Acc: {acc:.4f}")
        
            if acc > best_val_acc:
                best_val_acc = acc
                best_model_state = model.state_dict()
                best_params = {'lr': lr, 'dropout': dropout, 'hidden_size': hidden_size}
        
        # Store the best model
        self.model = Classifier(input_number, best_params['hidden_size'], best_params['dropout'], output_number).to(device)
        self.model.load_state_dict(best_model_state)
        self.model.eval()

        # Save the best model into a file
        if model_output_path:
            Best_Result = {
            'model_state': best_model_state,
            'best_params': best_params,
            'input_number': input_number,
            'output_number': output_number,   
            'best_val_acc':best_val_acc}
            os.makedirs(model_output_path, exist_ok=True)
            
            torch.save(Best_Result, os.path.join(model_output_path,model_output_name) )
        print('Model trained', 'and saved!' if model_output_path else '!')

    # Load the model from saved file
    def load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        hidden_size = checkpoint['best_params']['hidden_size']
        dropout = checkpoint['best_params']['dropout']
        input_number = checkpoint['input_number']
        output_number = checkpoint['output_number']
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = Classifier(input_number, hidden_size, dropout, output_number).to(device)
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        
        self.model = model

    
    # Prediction (return probability for each class)
    def predict_probabilities(self, df):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        embeddings = np.array(df['Embedding'].tolist())
        inputs = torch.tensor(embeddings, dtype=torch.float32).to(device)
    
        with torch.no_grad():
            outputs = self.model(inputs)
            probs = F.softmax(outputs, dim=1)
    
        return probs.cpu().tolist()  # List of [P(class_0), ..., P(class_n)] per row

    # Return class index with highest predicted probabilitiy
    def predict_class(self, embeddings):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        inputs = torch.tensor(embeddings, dtype=torch.float32).to(device)
    
        with torch.no_grad():
            outputs = self.model(inputs)
            probs = F.softmax(outputs, dim=1)
            
        return [int(torch.argmax(p)) for p in probs]

    # Validate the model
    def validate(self, df):
        y_true = df['Cat_Index'].tolist()
        y_pred = self.predict_class(df)

        acc = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred)

        print(f"Validation Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(report)
        return acc, report
