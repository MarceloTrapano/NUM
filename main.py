import train as tr
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch import seed_everything
from torchvision import models
import sys
import os

# RESIZE_SIZE = 256
RESIZE_SIZE = 224
CROP_SIZE = 224
DEFAULT_BATCH_SIZE = 20
DEFAULT_NUM_WORKERS = 10
DEFAULT_MAX_EPOCHS = 100
NUM_CLASSES = 5

def main(batch_size = DEFAULT_BATCH_SIZE, num_workers = DEFAULT_NUM_WORKERS, max_epochs = DEFAULT_MAX_EPOCHS, model_path = None, test_only = False):
    # mniejszy rozmiar - szybsze uczenie; resize -> crop by zachować detale na środku - brzegi mniej ważne
    # transform = tr.transforms.Compose([
    #     tr.transforms.Resize(RESIZE_SIZE),
    #     tr.transforms.CenterCrop(CROP_SIZE),
    #     tr.transforms.ToTensor()
    #     ])
    transform = tr.transforms.Compose([
        tr.transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
        tr.transforms.ToTensor()
        ])
    seed_everything(121, workers=True) # dla odtwarzalności uczenia
    # dataset: https://www.kaggle.com/datasets/imsparsh/flowers-dataset/
    data = tr.LightingData(
        dataset_dir = 'C:\\Users\\jkond\\Desktop\\studia\\NUM\\Dataset',
        batch_size = batch_size,
        num_workers = num_workers,
        transform = transform
    )
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='best-checkpoint'
    )
    csv_logger = CSVLogger("logs", name="test")
    wandb_logger = WandbLogger(project="pytorch-lightning")
    loggers = [csv_logger, wandb_logger]
    # model_type = models.resnet50(num_classes=num_classes)
    model_type = models.efficientnet_b1(num_classes=NUM_CLASSES)

    if model_path:
        model = tr.LightingModel.load_from_checkpoint(model_path, model = model_type)
    else:
        model = tr.LightingModel(lr=1e-3, model = model_type)
        
    if test_only == True:
        trainer = L.Trainer(logger = None)
        trainer.test(model, data)
        return
    
    trainer = L.Trainer(max_epochs = max_epochs, logger = loggers, callbacks=[checkpoint_callback], deterministic=True, val_check_interval=0.2)
    trainer.fit(model, data)
    trainer.test(model, data)

if __name__ == '__main__':
    model_path = None
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        model_path = sys.argv[1]
        print("Loading model from file.")
    if len(sys.argv) > 2 and sys.argv[2] == "test":
        test_only = True
    else:
        test_only = False
    main(model_path = model_path, test_only = test_only)