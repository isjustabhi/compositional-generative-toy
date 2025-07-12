from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class ShapeColorDataset(Dataset):
    def __init__(self, folder_path):
        self.folder = folder_path
        self.files = [f for f in os.listdir(folder_path) if f.endswith(".png")]
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()  # Converts to range [0, 1]
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder, self.files[idx])
        img = Image.open(img_path).convert("RGB")
        return self.transform(img)
