import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class ExampleDataSet0(Dataset):
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
    
class ExampleDataSet(Dataset):
    def __init__(self, dataset_dir, train = True, transform = None):
        self.images = []
        self.labels = []
        self.dataset_dir = dataset_dir
        self.set_type = 'train' if train else 'test'
        self.transform = transform
        self.class_map = {}
        path = os.path.join(dataset_dir, self.set_type)
        curr_class = 0
        class_dirs = sorted([cd for cd in os.listdir(path) if os.path.isdir(os.path.join(path, cd))])
        for class_dir in class_dirs:
            class_images = [img for img in os.listdir(os.path.join(path, class_dir)) if img.endswith('.jpg')]
            self.images.extend(class_images)
            self.labels.extend([class_dir] * len(class_images))
            self.class_map[class_dir] = curr_class
            curr_class += 1
            

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        label = self.labels[idx]
        img_path = os.path.join(self.dataset_dir, self.set_type, label, self.images[idx])
        label = self.class_map[label] # string -> int
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
