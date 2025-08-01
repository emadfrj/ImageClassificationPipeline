# Finding Anomalies in images using autoencoder recunstruction error

import torch
import torch.nn as nn
import numpy as np
import os

class Anomaly(nn.Module):
    def __init__(self,input_dim, device='cuda'):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent)

    def update_training_data(self, new_X_train):
        self.input_dim = len(new_X_train[0])
        self.inputs = torch.tensor(new_X_train, dtype=torch.float32).to(self.device)

    # Training function
    def train_autoencoder(self, X_train,model_output_path = None,model_output_name ='Autoencoder_Model.pth', epochs=50, lr=1e-3):
        inputs = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        self.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        self.train()

        for epoch in range(epochs):
            outputs = self(inputs)
            loss = criterion(outputs, inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")
        
        if model_output_path:
            os.makedirs(model_output_path, exist_ok=True)
            # save the model to file
            torch.save(self.state_dict(), os.path.join(model_output_path,model_output_name))
        print('Model trained', 'and saved!' if model_output_path else '!')

    def anomaly_detecting(self, X, threshold=None):
        self.eval()
        self.to(self.device)
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            recon = self(X_tensor)
            errors = ((X_tensor - recon) ** 2).mean(dim=1).cpu().numpy()

        if threshold is None:
            threshold = np.percentile(errors, 95)  # Top 5% as anomalies
        return (errors > threshold).astype(int), errors
    @staticmethod
    def load_model(input_dim, model_path, device='cuda'):
        model = Anomaly(input_dim=input_dim, device=device)
        state_dict = torch.load(model_path, map_location=device)
        
        # Load encoder and decoder weights
        model.encoder.load_state_dict({k.replace("encoder.", ""): v for k, v in state_dict.items() if k.startswith("encoder.")})
        model.decoder.load_state_dict({k.replace("decoder.", ""): v for k, v in state_dict.items() if k.startswith("decoder.")})
        
        model.to(device)
        model.eval()
        print(f"Model loaded and weights assigned from: {model_path}")
        return model

