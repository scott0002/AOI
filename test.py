from __future__ import print_function, division

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
plt.ion() 

def load_image():
    data_transforms = {
        'evaluation': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    data_dir = "LCC_FASD"
    image_datasets =datasets.ImageFolder(os.path.join(data_dir, 'evaluation'),
                                            data_transforms['evaluation'])
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=4,
                                                shuffle=True, num_workers=4)
    dataset_sizes = len(image_datasets) 
    class_names = image_datasets.classes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
    return dataset_sizes, device, class_names, dataloaders

def run_test():
    dataset_sizes, device, class_names, dataloaders=load_image()
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model.fc = nn.Linear(num_ftrs, 2)

    
    model.load_state_dict(torch.load('model.pkl'))
    model = model.to(device)
    running_loss = 0.0
    running_corrects = 0
    model.eval()
    #phase=['evalution']
    inputs = []
    labels = []
    curt = time.time()
    for a, b in dataloaders:
        if(time.time()-curt>5):
            print('loss', running_loss)
            print('corrects', running_corrects)
            curt = time.time()
        inputs = a.to(device)
        labels = b.to(device)
        criterion = nn.CrossEntropyLoss()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data)
    # statistics
    
    loss = running_loss / dataset_sizes
    acc = running_corrects.double() / dataset_sizes

    print('{} Loss: {:.4f} Acc: {:.4f}'.format('evaluation', loss, acc))

if __name__ == '__main__':
    run_test()