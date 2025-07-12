import torch
from torchvision.utils import save_image, make_grid
from vae_model import ConvVAE
from vae_dataset import ShapeColorDataset
from torch.utils.data import DataLoader
import os

# Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 32
BATCH_SIZE = 16

# Load trained model
model = ConvVAE(latent_dim=LATENT_DIM).to(DEVICE)
model.load_state_dict(torch.load("models/vae_baseline.pth", map_location=DEVICE))
model.eval()

# Load test data (including held-out combinations)
dataset = ShapeColorDataset("toy_dataset/test")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Create output folder
os.makedirs("samples/reconstructions", exist_ok=True)
os.makedirs("samples/generated", exist_ok=True)

# 🔁 Step 1: Reconstruct real images
with torch.no_grad():
    for i, batch in enumerate(dataloader):
        batch = batch.to(DEVICE)
        recon, _, _ = model(batch)

        # Save originals and reconstructions
        save_image(batch, f"samples/reconstructions/orig_{i}.png", nrow=4)
        save_image(recon, f"samples/reconstructions/recon_{i}.png", nrow=4)
        if i == 2:
            break  # just a few examples

# 🎲 Step 2: Generate images from random latent vectors
with torch.no_grad():
    z = torch.randn(16, LATENT_DIM).to(DEVICE)
    samples = model.decode(z)
    save_image(samples, "samples/generated/sampled.png", nrow=4)

print("✅ Samples saved in 'samples/' folder")
