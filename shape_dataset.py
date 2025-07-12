from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class GrayscaleShapeDataset(Dataset):
    def __init__(self, folder_path):
        self.folder = folder_path
        self.files = [f for f in os.listdir(folder_path) if f.endswith(".png")]
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),           # shape: [1, 64, 64]
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder, self.files[idx])
        img = Image.open(img_path).convert("L")  # Grayscale
        return self.transform(img)
    