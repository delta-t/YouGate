import os
import time
import numpy as np
from PIL import Image

from torch import nn
from torch.autograd import Variable
from torchvision import models
from torchvision import transforms


ALPHABET = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "A", "B", "C", "E", "H", "K", "M", "O", "P", "T", "X", "Y", "_"]
MAX_NOMER = 9
LENGTH_ALPHABET = len(ALPHABET)


def get_model():
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(in_features=512, out_features=LENGTH_ALPHABET*MAX_NOMER, bias=True)
    return model

def predict(model, image_path, image_name):
    """Function to ocr prediction"""
    start_time = time.perf_counter()
    
    model.cuda()
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
    ])

    image = Image.open(os.path.join(image_path, image_name))
    image = image.convert('L')
    image = transform(image)
    image = Variable(image).cuda()
    image = image.unsqueeze(dim=1)
    pred = model(image)

    c0 = ALPHABET[np.argmax(pred.squeeze().cpu().tolist()[0:LENGTH_ALPHABET])]
    c1 = ALPHABET[np.argmax(pred.squeeze().cpu().tolist()[LENGTH_ALPHABET:LENGTH_ALPHABET*2])]
    c2 = ALPHABET[np.argmax(pred.squeeze().cpu().tolist()[LENGTH_ALPHABET*2:LENGTH_ALPHABET*3])]
    c3 = ALPHABET[np.argmax(pred.squeeze().cpu().tolist()[LENGTH_ALPHABET*3:LENGTH_ALPHABET*4])]
    c4 = ALPHABET[np.argmax(pred.squeeze().cpu().tolist()[LENGTH_ALPHABET*4:LENGTH_ALPHABET*5])]
    c5 = ALPHABET[np.argmax(pred.squeeze().cpu().tolist()[LENGTH_ALPHABET*5:LENGTH_ALPHABET*6])]
    c6 = ALPHABET[np.argmax(pred.squeeze().cpu().tolist()[LENGTH_ALPHABET*6:LENGTH_ALPHABET*7])]
    c7 = ALPHABET[np.argmax(pred.squeeze().cpu().tolist()[LENGTH_ALPHABET*7:LENGTH_ALPHABET*8])]
    c8 = ALPHABET[np.argmax(pred.squeeze().cpu().tolist()[LENGTH_ALPHABET*8:LENGTH_ALPHABET*9])]

    number_plate = '%s%s%s%s%s%s%s%s%s' % (c0, c1, c2, c3, c4, c5, c6, c7, c8)
    number_plate =  number_plate.replace('_','')
    end_time = time.perf_counter()

    return number_plate, end_time - start_time