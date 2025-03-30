import bentoml
import os
from PIL import Image
from torchvision import transforms
from utils import PredictDataSet
from torch.utils.data import DataLoader

transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
        ])

prdecict = PredictDataSet("NUM/Dataset/predict", transform=transform)
predict_dataset = DataLoader(prdecict, num_workers=4, batch_size = 16)
CNN_runner = bentoml.pytorch_lightning.get("CNN:latest").to_runner()
CNN_runner.init_local()


for batch in predict_dataset:
    print(CNN_runner.run(batch))

