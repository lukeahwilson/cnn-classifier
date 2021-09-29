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



def c1_download_pretrained_model(model_name):
    #Download a pretrained convolutional neural network to reference
    models = {'vgg': vgg16, 'alexnet': alexnet, 'googlenet': googlenet, 'densenet': densenet,
              'inception': inception, 'resnext50': resnext50, 'shufflenet': shufflenet}
    model = models[model_name]
    #Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    pretrained_output_name = list(model._modules.items())[-1][0]
    model._modules['new_output'] = model._modules.pop(pretrained_output_name)
    in_features = model.new_output.weight.shape[1]
    out_features = len(train_data.classes)


def c2_create_classifier(model, in_features, out_features):
    #Create a Classifier class, inheriting from nn.Module and incorporating Relu, Dropout and log_softmax
    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(in_features, 512)
            self.fc2 = nn.Linear(512, out_features)
            self.dropout = nn.Dropout(p=0.3)

        def forward(self, x):
            x = x.view(x.shape[0], -1)
            x = self.dropout(F.relu(self.fc1(x)))
            x = F.log_softmax(self.fc2(x), dim=1)
            return x

def
#    Create dictionary of all these variables for training
    #Initialize testing loss history
    training_loss_history, validate_loss_history, testing_loss_history, overfit_loss_history = [], [], [], []

    #Initialize tracker for activating CNN layers
    running_count = 0


    #Define the criterion
    criterion = nn.NLLLoss()

    #Replace the fully connected layer(s) at the end of the model with our own fully connected classifier
    model.new_output = Classifier()

    #Define learning rate and weight decay
    learnrate=0.003
    weightdecay=0.00001
    startlearn=learnrate

    # Only train the classifier (fc) parameters, feature parameters are frozen
    optimizer = optim.Adam(model.fc.parameters(), lr=learnrate, weight_decay=weightdecay)
