from __future__ import print_function, division
import pandas as pd
import torch
import math
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torchvision.io import read_image
plt.ion() 

def run_test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
    data_transforms = {
        'evaluation': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, ], [0.229, ])
        ]),
    }
    img_dir = "./aoi/test_images"
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model.fc = nn.Linear(num_ftrs, 6)

    
    model.load_state_dict(torch.load('./model.pkl'))
    model = model.to(device)
    #phase=['evalution']
    predict = []
    img_labels = pd.read_csv("./aoi/test.csv")
    for idx in range(len(img_labels)):
        img_path = os.path.join(img_dir, img_labels.iloc[idx, 0])
        image = read_image(img_path)
        inputs = data_transforms['evaluation'](image.repeat(3, 1, 1)).unsqueeze(0).to(device)
        outputs = model(inputs)
        _, pred = torch.max(outputs, 1)
        # print(int(pred[0]))
        predict.append(int(pred[0]))
    submit = pd.DataFrame({'ID': img_labels.iloc[:, 0],
                            'Label': predict})
    submit.to_csv("./aoi/submit.csv",
                    header=True,
                    sep=',',
                    encoding='utf-8',
                    index=False)
if __name__ == '__main__':
    run_test()