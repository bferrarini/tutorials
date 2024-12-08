import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import copy
import numpy as np

# Definizione dell'autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder
        self.enc1 = nn.Linear(784, 128)
        self.enc2 = nn.Linear(128, 96)
        self.enc3 = nn.Linear(96, 64)
        self.enc4 = nn.Linear(64, 24)
        self.enc5 = nn.Linear(24, 3)

        # Decoder 
        self.dec1 = nn.Linear(3, 24)
        self.dec2 = nn.Linear(24, 64)
        self.dec3 = nn.Linear(64, 96)
        self.dec4 = nn.Linear(96, 128)
        self.dec5 = nn.Linear(128, 784)

    def forward(self, x):
        # Encoder
        x = torch.relu(self.enc1(x))
        x = torch.relu(self.enc2(x))
        x = torch.relu(self.enc3(x))
        x = torch.relu(self.enc4(x))
        x = self.enc5(x) # Codice latente

        # Decoder
        x = torch.relu(self.dec1(x))
        x = torch.relu(self.dec2(x))
        x = torch.relu(self.dec3(x))
        x = torch.relu(self.dec4(x))
        x = torch.sigmoid(self.dec5(x)) # Output ricostruito
        return x

import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder
        self.enc1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)  # 1x28x28 -> 16x14x14
        self.enc2 = nn.ReLU()
        self.enc3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # 16x14x14 -> 32x7x7
        self.enc4 = nn.ReLU()
        self.enc5 = nn.Conv2d(32, 64, kernel_size=7)  # 32x7x7 -> 64x1x1

        # Decoder
        self.dec1 = nn.ConvTranspose2d(64, 32, kernel_size=7)  # 64x1x1 -> 32x7x7
        self.dec2 = nn.ReLU()
        self.dec3 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)  # 32x7x7 -> 16x14x14
        self.dec4 = nn.ReLU()
        self.dec5 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)  # 16x14x14 -> 1x28x28
        self.dec6 = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)

        # Decoder
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.dec5(x)
        x = self.dec6(x)
        return x

# Iperparametri
epochs = 10
batch_size = 32
learning_rate = 0.9e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Caricamento del dataset MNIST
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, 
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True)

# Inizializzazione del modello, ottimizzatore e funzione di loss
# model = Autoencoder()
model = ConvAutoencoder()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Training dell'autoencoder
for epoch in range(epochs):
    for batch_idx, (data, _) in enumerate(train_loader):
        # Appiattimento delle immagini
        if isinstance(model,Autoencoder):
            data = data.view(-1, 784)
        
        # Forward pass
        data = data.to(device)
        output = model(data)
        loss = criterion(output, data)
        
        # Backward pass e ottimizzazione
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, epochs, batch_idx, len(train_loader), loss.item()))

# Test dell'autoencoder e visualizzazione
with torch.no_grad():
    for data, _ in test_loader:
        if isinstance(model,Autoencoder):
            data = data.view(-1, 784)
        data = data.to(device)
        output = model(data)
        
        D = data.cpu()
        O = output.cpu()

        # Visualizzazione delle immagini originali e ricostruite
        for i in range(5):
            # Immagine originale
            plt.subplot(2, 5, i + 1)
            plt.imshow(D[i].view(28, 28), cmap='gray')
            plt.title('Originale')

            # Immagine ricostruita
            plt.subplot(2, 5, i + 6)
            plt.imshow(O[i].view(28, 28), cmap='gray')
            plt.title('Ricostruita')

        plt.show()
        break  # Mostra solo un batch di immagini