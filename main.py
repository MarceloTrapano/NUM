import train as tr
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch import seed_everything
import sys
import os

RESIZE_SIZE = 256
CROP_SIZE = 224
DEFAULT_BATCH_SIZE = 6
DEFAULT_NUM_WORKERS = 3
DEFAULT_MAX_EPOCHS = 2

def main(batch_size = DEFAULT_BATCH_SIZE, num_workers = DEFAULT_NUM_WORKERS, max_epochs = DEFAULT_MAX_EPOCHS, model_path = None, aha):
    # mniejszy rozmiar - szybsze uczenie; resize -> crop by zachować detale na środku - brzegi mniej ważne
    transform = tr.transforms.Compose([
        tr.transforms.Resize(RESIZE_SIZE),
        tr.transforms.CenterCrop(CROP_SIZE),
        tr.transforms.ToTensor()
        ])
    # dataset: https://www.kaggle.com/datasets/stealthknight/bird-vs-drone
    data = tr.LightingData(
        dataset_dir = 'C:\\Users\\Cymentio\\Desktop\\studia\\narzędzia uczenia maszynowego\\zad1\\Dataset',
        batch_size = batch_size,
        num_workers = num_workers,
        transform = transform
    )
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='my/path/',
        filename='best-checkpoint'
    )
    logger = CSVLogger("logs", name="test")
    seed_everything(42, workers=True) # dla odtwarzalności uczenia

    if model_path:
        model = tr.LightingModel.load_from_checkpoint(model_path, num_classes = 2)
        trainer = L.Trainer(logger = logger)
        trainer.test(model, data)
        return
    else:
        model = tr.LightingModel(num_classes = 2, lr=1e-5)
    
    trainer = L.Trainer(max_epochs = max_epochs, logger = logger, callbacks=[checkpoint_callback], deterministic=True, val_check_interval=0.2, limit_train_batches=0.25, limit_val_batches=0.25)
    trainer.fit(model, data)
    trainer.test(model, data)

if __name__ == '__main__':
    model_path = None
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        model_path = sys.argv[1]
        print("Loading model from file.")
    main(model_path = model_path)