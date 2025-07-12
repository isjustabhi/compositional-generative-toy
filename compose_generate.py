import torch
from torchvision.utils import save_image
from shape_vae_model import ConvVAE
from color_mapper import ColorMapper
import os

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 16
NUM_SAMPLES = 16

# Load models
shape_vae = ConvVAE(latent_dim=LATENT_DIM).to(DEVICE)
shape_vae.load_state_dict(torch.load("models/shape_vae.pth", map_location=DEVICE))
shape_vae.eval()

color_mapper = ColorMapper().to(DEVICE)
color_mapper.load_state_dict(torch.load("models/color_mapper.pth", map_location=DEVICE))
color_mapper.eval()

# Generate grayscale shapes from VAE
with torch.no_grad():
    z = torch.randn(NUM_SAMPLES, LATENT_DIM).to(DEVICE)
    gray_shapes = shape_vae.decode(z)  # Output: (N, 1, 64, 64)

# Use color mapper to colorize
with torch.no_grad():
    colored_shapes = color_mapper(gray_shapes)  # Output: (N, 3, 64, 64)

# Save outputs
os.makedirs("samples/composed", exist_ok=True)
save_image(gray_shapes, "samples/composed/grayscale.png", nrow=4)
save_image(colored_shapes, "samples/composed/colored.png", nrow=4)

print("✅ Compositional generation complete. Check samples/composed/")
