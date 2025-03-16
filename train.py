import torch
from torchvision import transforms, datasets, models
import lightning as L
from data import ExampleDataSet
from torch.utils.data import DataLoader, random_split


class LightingModel(L.LightningModule):
    def __init__(self, model, lr = 1e-3, loss_fn = torch.nn.CrossEntropyLoss()):
        super().__init__()
        self.save_hyperparameters(ignore=['loss_fn', 'model'])
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        images, targets = batch
        outputs = self.model(images)
        loss = self.loss_fn(outputs, targets)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch):
        images, targets = batch
        outputs = self.model(images)
        loss = self.loss_fn(outputs, targets)
        accuracy = (outputs.argmax(dim=1) == targets).float().mean()
        self.log('val_acc', accuracy)
        self.log('val_loss', loss)

    def test_step(self, batch):
        images, targets = batch
        outputs = self.model(images)
        loss = self.loss_fn(outputs, targets)
        accuracy = (outputs.argmax(dim=1) == targets).float().mean()
        self.log('test_acc', accuracy)
        self.log('test_loss', loss)


    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
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
        transform = self.transform
        train_dataset = self.train_dataset
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)
    
    def val_dataloader(self):
        transform = self.transform
        val_dataset = self.val_dataset
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)
    
    def test_dataloader(self):
        transform = self.transform
        test_dataset = ExampleDataSet(self.dataset_dir, train=True, transform=transform)
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)