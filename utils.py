import os
import torch
import pytorch_lightning as L
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
from PIL import Image
from torch.nn import Conv2d, Linear, MaxPool2d, ReLU, Dropout, Sequential, Flatten

RESIZE_SIZE = 128
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 4
DEFAULT_MAX_EPOCHS = 10
NUM_CLASSES = 5

class PredictDataSet(Dataset):
    def __init__(self, dataset_dir, transform = None):
        self.dataset_dir = dataset_dir
        self.transform = transform
        path = os.path.join(dataset_dir)
        self.images = [img for img in os.listdir(path) if img.endswith('.jpg')]
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

class ExampleDataSet(Dataset):
    def __init__(self, dataset_dir, train = True, transform = None):
        self.images = []
        self.labels = []
        self.dataset_dir = dataset_dir
        self.set_type = 'train' if train else 'test'
        self.transform = transform
        self.class_map = {}
        self.train = train
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

class LightingData(L.LightningDataModule):
    def __init__(self, dataset_dir, batch_size, num_workers, transform = transforms.ToTensor(), train_ratio = 0.8):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        training_dataset = ExampleDataSet(self.dataset_dir, train=True, transform=transform)
        train_size = int(len(training_dataset) * train_ratio)
        val_size = len(training_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(training_dataset, [train_size, val_size])
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)
    
    def test_dataloader(self):
        test_dataset = ExampleDataSet(self.dataset_dir, train=False, transform=self.transform)
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)
    def predict_dataloader(self):
        predict_dataset = ExampleDataSet(self.dataset_dir, train=False, transform=self.transform)
        return DataLoader(predict_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)
def calculate_num_of_nodes(size):
    value = size - 2
    value //= 2
    value -= 2
    value //= 2
    return value*value*64*3

class LightingModel(L.LightningModule):
    def __init__(self, lr = 0.0002137, loss_fn = torch.nn.CrossEntropyLoss(), num_classes = NUM_CLASSES, dropout = 0.43, linear_nodes = 99):
        super().__init__()
        self.save_hyperparameters(ignore=['loss_fn', 'model'])

        self.linear1 = Linear(calculate_num_of_nodes(RESIZE_SIZE), linear_nodes)
        self.conv1 = Conv2d(in_channels=3, out_channels=32*3,kernel_size=(3, 3)) 
        self.conv2 = Conv2d(in_channels=32*3, out_channels=64*3,kernel_size=(3, 3)) 
        self.maxpool = MaxPool2d(kernel_size=(2, 2)) #
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(linear_nodes, num_classes)

        self.num_classes = num_classes
        self.loss_fn = loss_fn
        self.lr = lr

        self.cnn = Sequential(
            self.conv1,
            self.maxpool,
            ReLU(),
            self.conv2,
            self.maxpool,
            ReLU(),
            Flatten(),
            self.linear1,
            ReLU(),
            self.dropout,
            self.linear2,
        )

    def forward(self, x):
        return self.cnn(x)
    def training_step(self, batch):
        images, targets = batch
        outputs = self(images)
        loss = self.loss_fn(outputs, targets)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch):
        images, targets = batch
        outputs = self(images)
        loss = self.loss_fn(outputs, targets)
        accuracy = (outputs.argmax(dim=1) == targets).float().mean()
        self.log('val_acc', accuracy)
        self.log('val_loss', loss)

    def test_step(self, batch):
        images, targets = batch
        outputs = self(images)
        loss = self.loss_fn(outputs, targets)
        accuracy = (outputs.argmax(dim=1) == targets).float().mean()
        self.log('test_acc', accuracy)
        self.log('test_loss', loss)

    def predict_step(self, batch):
        return self(batch)


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)