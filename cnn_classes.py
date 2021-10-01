#!/usr/bin/python
# PROGRAMMER: Luke Wilson
# DATE CREATED: 2021-09-27
# REVISED DATE: 2021-09-28
# PURPOSE:
#   - Provide utility functions for import into main
#       o c1_download_pretrained_model for downloading desireable pretrained models to use
#       o c2_create_classifier() for reassigning output layers to newly attached layers and creating initialized classifier
##

# Import required libraries
import time, os, random
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets, models
from torch import nn, optim
from PIL import Image
import matplotlib.pyplot as plt
%matplotlib inline


vgg16 = models.vgg16()
alexnet = models.alexnet()
googlenet = models.googlenet()
densenet = models.densenet161()
inception = models.inception_v3()
resnext50 = models.resnext50_32x4d()
shufflenet = models.shufflenet_v2_x1_0()


#Create a Classifier class, inheriting from nn.Module and incorporating Relu, Dropout and log_softmax
class Classifier(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc1 = nn.Linear(self.in_features, 512)
        self.fc2 = nn.Linear(512, self.out_features)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x


def m1_download_pretrained_model(model_name):
    #Download a pretrained convolutional neural network to reference
    models = {'vgg': vgg16, 'alexnet': alexnet, 'googlenet': googlenet, 'densenet': densenet,
              'inception': inception, 'resnext50': resnext50, 'shufflenet': shufflenet}
    model = models[model_name]

    #Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    pretrained_output_name = list(model._modules.items())[-1][0]
    model._modules['new_output'] = model._modules.pop(pretrained_output_name)
    return model


def m2_create_classifier(train, model_name, classes_length):
    model = c1_download_pretrained_model(model_name):

    in_features = model.new_output.weight.shape[1]
    out_features = classes_length

    #Replace the fully connected layer(s) at the end of the model with our own fully connected classifier
    model.new_output = Classifier(in_features, out_features)

    #Define the criterion
    criterion = nn.NLLLoss()

    #Define learning rate and weight decay
    learnrate=0.003
    weightdecay=0.00001
    startlearn=learnrate

    # Only train the classifier (fc) parameters, feature parameters are frozen
    optimizer = optim.Adam(model.fc.parameters(), lr=learnrate, weight_decay=weightdecay)

    if train:
        model = c1_download_pretrained_model(model_name)
    if not train:
        m4_save_model_checkpoint(model, learnrate, training_loss_history, validate_loss_history, epoch_on, running_count)
        model = c
    model =


#    Create dictionary of all these variables for training
    #Initialize testing loss history
    training_loss_history, validate_loss_history, testing_loss_history, overfit_loss_history = [], [], [], []

    #Initialize tracker for activating CNN layers
    running_count = 0


def m3_load_model_checkpoint():
    # Option to reload from previous state
    checkpoint = torch.load('flower-classifier-googlenet-dict.pth')
    model.load_state_dict(checkpoint)

    with open(file_hyperparameters, 'r') as file:
        model_hyperparameters = json.load(file)

    learnrate = model_hyperparameters['learnrate']
    training_loss_history = model_hyperparameters['training_loss_history']
    validate_loss_history = model_hyperparameters['validate_loss_history']
    epoch_on = model_hyperparameters['epoch_on']
    running_count = model_hyperparameters['running_count']



"""
    print('loaded model learnrate = ', learnrate)
    plt.plot(training_loss_history, label='Training Training Loss')
    plt.plot(validate_loss_history, label='Validate Training Loss')
    plt.vlines(
        colors = 'black',
        x = epoch_on,
        ymin = min(training_loss_history),
        ymax = max(training_loss_history[5:]),
        linestyles = 'dotted',
        label = 'CNN Layers Activated'
    ).set_clip_on(False)
    plt.vlines(
        colors = 'black',
        x = epoch_on + running_count,
        ymin = min(training_loss_history),
        ymax = max(training_loss_history[5:]),
        linestyles = 'dotted',
        label = 'CNN Layers Deactivated'
    ).set_clip_on(False)
    plt.ylabel('Total Loss')
    plt.xlabel('Total Epoch ({})'.format(len(training_loss_history)))
    plt.legend(frameon=False)
"""


def m4_save_model_checkpoint(model, learnrate, training_loss_history, validate_loss_history, epoch_on, running_count):
    #Save the model state_dict
    torch.save(model.state_dict(), 'flower-classifier-googlenet-dict.pth')
    model.fc.state_dict().keys()
    file_hyperparameters = 'flower-classifier-googlenet-hyperparameters.json'
    #Save the model hyperparameters and the locations in which the CNN training activated and deactivated
    model_hyperparameters = {'learnrate':learnrate,
                             'training_loss_history': training_loss_history,
                             'validate_loss_history': validate_loss_history,
                             'epoch_on': epoch_on,
                             'running_count': running_count}

    #Create a JSON file containing the saved information above
    with open(file_hyperparameters, 'w') as file:
        json.dump(model_hyperparameters, file)
