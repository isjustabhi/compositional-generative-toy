import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from color_dataset import ColorMappingDataset
from color_mapper import ColorMapper
from torchvision.utils import save_image
import os

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50
BATCH_SIZE = 32
SAMPLE_EVERY = 5  # Save visual outputs every N epochs

# Dataset and DataLoader
dataset = ColorMappingDataset("toy_dataset/train")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model and optimizer
model = ColorMapper().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Create output directory
os.makedirs("models", exist_ok=True)
os.makedirs("samples/progress", exist_ok=True)

# Training loop
for epoch in range(EPOCHS):
    total_loss = 0
    model.train()

    for x_gray, y_rgb in dataloader:
        x_gray, y_rgb = x_gray.to(DEVICE), y_rgb.to(DEVICE)
        output = model(x_gray)

        loss = F.mse_loss(output, y_rgb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {total_loss:.4f}")

    # 🔍 Save predictions every 5 epochs
    # 🔍 Save predictions every 5 epochs
    if (epoch + 1) % SAMPLE_EVERY == 0 or epoch == EPOCHS - 1:
        print(f"→ Saving sample images for epoch {epoch+1}")
        model.eval()
        with torch.no_grad():
            val_input = x_gray[:8]  # First 8 from last batch
            val_output = model(val_input)

            save_image(val_input, f"samples/progress/input_gray_epoch{epoch+1}.png", nrow=4)
            save_image(val_output, f"samples/progress/output_color_epoch{epoch+1}.png", nrow=4)
        model.train()


# Save final model
torch.save(model.state_dict(), "models/color_mapper.pth")
print("✅ Done. Model saved and progress images generated.")
