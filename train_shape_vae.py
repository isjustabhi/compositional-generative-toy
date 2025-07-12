import torch
from torch.utils.data import DataLoader
from shape_dataset import GrayscaleShapeDataset
from shape_vae_model import ConvVAE
import torch.nn.functional as F
import os

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 20
LATENT_DIM = 16

# Load data
dataset = GrayscaleShapeDataset("grayscale_shapes/train")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model
model = ConvVAE(latent_dim=LATENT_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Loss
def vae_loss(x_recon, x, mu, logvar):
    recon = F.mse_loss(x_recon, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl

# Training loop
for epoch in range(EPOCHS):
    total_loss = 0
    for batch in dataloader:
        batch = batch.to(DEVICE)
        x_recon, mu, logvar = model(batch)
        loss = vae_loss(x_recon, batch, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.2f}")

# Save
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/shape_vae.pth")
print("✅ Shape VAE trained and saved.")
