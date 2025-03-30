import bentoml
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from bentoml.io import Image, Text

cnn_runner = bentoml.pytorch_lightning.get("CNN:latest").to_runner()

cnn = bentoml.Service("Image_classifier", runners=[cnn_runner])

@cnn.api(input=Image(), output=Text())
def classify(image):
    classification = ["Daisy","Dandelion","Rose","Sunflower","Tulip"]
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
        ])
    image = transform(image)

    predict_dataset = DataLoader([image])
    for batch in predict_dataset:
        return classification[np.argmax(cnn_runner.run(batch).numpy())]