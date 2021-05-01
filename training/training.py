from __future__ import annotations, print_function, division
import torch.utils.data as data
import torch
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
import pandas as pd
from torchvision.io import read_image
from  dataloader import CustomImageDataset as dataloader
from dataloader import DatasetFromSubset
plt.ion()   # interactive mode

def load_model():
    model = torch.hub.load('pytorch/vision:v0.6.0', 'googlenet', pretrained=True)
    model.eval()
    return model

def load_image():
    data_transforms = {
        'training': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'development': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    # create dataloader
    data_dir = "../aoi/train_images"
    annotations_file_path = "../aoi/train.csv"
    dataset = dataloader(annotations_file_path,data_dir)
    
    # split dataset into train & dev
    train_set_size = int(len(dataset) * 0.8)
    valid_set_size = len(dataset) - train_set_size
    train_set, valid_set = data.random_split(dataset, [train_set_size, valid_set_size])

    image_datasets = {'training': DatasetFromSubset(train_set,data_transforms['training'])
        , 'development': DatasetFromSubset(valid_set,data_transforms['development'])}
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=4)
                for x in ['training', 'development']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['training', 'development']}
    class_names = [0,1,2,3,4,5]
    # print(class_names)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
    return dataset_sizes, device, class_names, dataloaders

def imshow(class_names, dataloaders, inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(10)  # pause a bit so that plots are updated
    # Get a batch of training data

def train_model(class_names, dataset_sizes, dataloaders, device, model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['training', 'development']:
            if phase == 'training':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'training'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'training':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'training':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'development' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
def visualize_model(inp, device, class_names, dataloaders ,model, num_images=6, title=None):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['development']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(class_names, dataloaders, inp, title)

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

def pretrain_model_train(class_names, dataset_sizes, dataloaders,device):
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    model_ft = train_model(class_names, dataset_sizes, dataloaders, device, model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)
    return model_ft

def training():
    dataset_sizes, device, class_names, dataloaders=load_image()
    model_ft= pretrain_model_train(class_names, dataset_sizes, dataloaders,device)
    torch.save(model_ft.state_dict(), '../model.pkl')
    
    #inputs, classes = next(iter(dataloaders['training']))
    #out = torchvision.utils.make_grid(inputs)
    
    #visualize_model(out, device, class_names, dataloaders ,model_ft, num_images=6, title=[class_names[x] for x in classes])
    #model=load_model()
    #train_model(class_names, dataset_sizes, dataloaders, device, model, criterion, optimizer, scheduler, num_epochs=25)



if __name__ == '__main__':
    training()