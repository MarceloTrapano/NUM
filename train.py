import torch
from torchvision import transforms, datasets, models
import lightning as L
from data import ExampleDataSet
from torch.utils.data import DataLoader


class LightingModel(L.LightningModule):
    def __init__(self, num_classes, lr = 1e-3, loss_fn = torch.nn.CrossEntropyLoss()):
        super().__init__()
        self.save_hyperparameters()
        self.model = models.efficientnet_b0(num_classes=num_classes)
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
        self.log('val_acc', accuracy)
        self.log('val_loss', loss)


    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
class LightingData(L.LightningDataModule):
    def __init__(self, dataset_dir, batch_size, num_workers, transform = transforms.ToTensor()):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
    
    def train_dataloader(self):
        transform = self.transform
        train_dataset = ExampleDataSet(self.dataset_dir, set_type='train', transform=transform)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        transform = self.transform
        val_dataset = ExampleDataSet(self.dataset_dir, set_type='valid', transform=transform)
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self):
        transform = self.transform
        test_dataset = ExampleDataSet(self.dataset_dir, set_type='test', transform=transform)
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)