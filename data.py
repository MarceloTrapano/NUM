import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class ExampleDataSet(Dataset):
    def __init__(self, dataset_dir, set_type = 'train', transform = None):
        self.images = [x for x in os.listdir(os.path.join(dataset_dir, set_type, "images")) if x.endswith('.jpg')]
        self.labels = [0 if filename[0].lower() == "d" else 1 for filename in self.images]
        self.dataset_dir = dataset_dir
        self.set_type = set_type
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_dir, self.set_type, "images", self.images[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
