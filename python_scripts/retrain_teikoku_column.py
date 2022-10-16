# -*- coding: utf-8 -*-
"""
retrain_teikoku_column.py

Created on Mon Sep 19 10:48:49 2022

@author: Lukas
"""

# import packages
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import time
import os
import copy


# set path directories
img_dir_path = ''
save_path = ''
n_epochs = ''
resnet_backbone = 0


# auxilliary functions
def train_model(model, criterion, optimizer, 
                scheduler, dataloaders, device, 
                dataset_sizes, num_epochs=25):
    """
    [function description goes here]

    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    criterion : TYPE
        DESCRIPTION.
    optimizer : TYPE
        DESCRIPTION.
    scheduler : TYPE
        DESCRIPTION.
    dataloaders : TYPE
        DESCRIPTION.
    device : TYPE
        DESCRIPTION.
    dataset_sizes : TYPE
        DESCRIPTION.
    num_epochs : TYPE, optional
        DESCRIPTION. The default is 25.

    Returns
    -------
    model : TYPE
        DESCRIPTION.

    """
    # measure training time
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # train for num_epochs epochs
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
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
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            if phase == 'train':
                scheduler.step()

            # loss and accuracy for a given epoch
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    # return total time required for training process
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model


# main function
def main(data_dir, out_path, epochs, n_visualize, model_depth):
    """
    [function description goes here]

    Parameters
    ----------
    data_dir : TYPE
        DESCRIPTION.
    out_path : TYPE
        DESCRIPTION.
    epochs : TYPE
        DESCRIPTION.
    n_visualize : TYPE
        DESCRIPTION.
    model_depth : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(), # data augmentation?
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # create image dataset as dictionary with names as keys
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
    
    # load data in batches of size 4
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                 shuffle=True, num_workers=4)
                  for x in ['train', 'val']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    
    class_names = image_datasets['train'].classes

    # check if we can use a GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # lazily load the data
    inputs, classes = next(iter(dataloaders['train']))

    # check which resnet we're loading
    if model_depth == 101:
        model_ft = models.resnet101(pretrained=True)
    elif model_depth == 50:
        model_ft = models.resnet50(pretrained=True)
    elif model_depth == 18:
        model_ft = models.resnet18(pretrained=True)

    num_ftrs = model_ft.fc.in_features
    
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft,
                           exp_lr_scheduler, dataloaders=dataloaders,
                           device=device, num_epochs=epochs,
                           dataset_sizes=dataset_sizes)
    torch.save(model_ft.state_dict(), os.path.join(out_path, "model.pth"))


if __name__ == "__main__":
    main(data_dir = img_dir_path,
         out_path = save_path,
         epochs = n_epochs,
         n_visualize = 0,
         model_depth = resnet_backbone)