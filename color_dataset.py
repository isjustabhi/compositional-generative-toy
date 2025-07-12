from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class ColorMappingDataset(Dataset):
    def __init__(self, folder_path):
        self.folder = folder_path
        self.files = [f for f in os.listdir(folder_path) if f.endswith(".png")]

        self.to_gray = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Grayscale(),       # 1 channel input
            transforms.ToTensor()
        ])

        self.to_color = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()         # 3 channel output
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.folder, self.files[idx])
        img = Image.open(path).convert("RGB")

        x_gray = self.to_gray(img)     # input (1, 64, 64)
        y_rgb = self.to_color(img)     # target (3, 64, 64)

        return x_gray, y_rgb
