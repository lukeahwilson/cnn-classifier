#!/usr/bin/python
# PROGRAMMER: Luke Wilson
# DATE CREATED: 2021-09-27
# REVISED DATE: 2021-10-27
# PURPOSE: Provide model functions for import into main
#   - Classifier(nn.Module)
#   - m1_create_classifier(model_name, hidden_layers, classes_length)
#   - m2_save_model_checkpoint(model, file_name_scheme, model_hyperparameters)
#   - m3_load_model_checkpoint(model, file_name_scheme)
##

# Import required libraries
import json
import time, os, random
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets, models
from torch import nn, optim
from PIL import Image
import matplotlib.pyplot as plt


# Store a dictionary of available models as names to avoid downloading models until a choice has been made
model_name_dic = {'vgg': 'vgg16', 'alexnet': 'alexnet', 'googlenet': 'googlenet', 'densenet': 'densenet161',
                  'resnext': 'resnext50_32x4d', 'shufflenet': 'shufflenet_v2_x1_0'}


#Create a Classifier class, inheriting from nn.Module and incorporating Relu, Dropout and log_softmax
class Classifier(nn.Module):
    '''
    Inherits Class information from the nn.Module and creates a Classifier Class:
        - Class has these attributes:
            o fully connected layer with specified number of in_features and out_features
            o number of hidden layers equivalent to the inputted requirements
            o dropout parameter for the fully connected layers
        - Class has a forward method:
            o Flattens the input data in an input layer for computation
            o Connects each layer with a relu activation, the defined dropout, and linear regression
            o Returns outputs from the final hidden layer into an categorical output probability using log_softmax
    Parameters:
        - in_features
        - hidden_layers
        - out_features
    '''
    def __init__(self, in_features, hidden_layers, out_features):
        super().__init__()
        self.in_features = in_features
        self.hidden_layers = hidden_layers
        self.out_features = out_features
        self._index = 1
        while self._index < self.hidden_layers:
            setattr(self, 'fc'+str(self._index), nn.Linear(round(self.in_features/(2**(self._index-1))),
                            round(self.in_features/(2**self._index))))
            self._index += 1
        setattr(self, 'fc'+str(self._index), nn.Linear(round(self.in_features/(2**(self._index-1))),
                        self.out_features))
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        self._index = 1
        while self._index < self.hidden_layers:
            x = self.dropout(F.relu(getattr(self,'fc'+str(self._index))(x)))
            self._index += 1

        x = F.log_softmax(getattr(self,'fc'+str(self._index))(x), dim=1)
        return x


def m1_create_classifier(model_name, hidden_layers, classes_length):
    '''
    Purpose:
        - Create classifier functions
        - Leverages the requested pretrained model to provide base features
    Parameters:
        - model_name = base pretrained model
        - hidden_layers = number of hidden layers in final fully connected network
        - out_features = number of classes in data
    Returns:
        - model
    '''

    #Download a pretrained convolutional neural network to reference, choose only the model requested by the user
    model = getattr(models, model_name_dic[model_name])(pretrained=True)

    # Ensure that the in and out features for our model seamlessly match the in from the pretrained CNN and the out for the classes
    # Rename the pretrained output layer to a default name 'new_output'
    # pretrained_output_name = list(model._modules.items())[-1][0]
    # model._modules['new_output'] = model._modules.pop(pretrained_output_name)
    out_features = classes_length

    for module in list(model.modules()):
        if module._get_name() == 'Linear':
            in_features = module.weight.shape[1]
            break

    #Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    #Replace the fully connected layer(s) at the end of the model with our own fully connected classifier
    setattr(model, list(model._modules.items())[-1][0], Classifier(in_features, hidden_layers, out_features))
    print('\nUsing ', model_name, ' with the following attached ', hidden_layers,
                    ' layer classifier:\n', list(model.children())[-1])

    return model


def m2_save_model_checkpoint(model, file_name_scheme, model_hyperparameters):
    '''
    Purpose:
        - Save model checkpoint and hyperparameters
    Parameters:
        - model = model to be saved
        - file_name_scheme = directory and naming convention for saving
        - model_hyperparameters = information about state of model
    Returns:
        - none
    '''
    #Save the model state_dict
    torch.save(model.state_dict(), file_name_scheme + '_dict.pth')

    #Create a JSON file containing the saved information above
    with open(file_name_scheme + '_hyperparameters.json', 'w') as file:
        json.dump(model_hyperparameters, file)


def m3_load_model_checkpoint(model, file_name_scheme):
    '''
    Purpose:
        - Load model checkpoint and hyperparameters
    Parameters:
        - model = model to be loaded
        - file_name_scheme = directory and naming convention for loading
    Returns:
        - model
        - model hyperparameters
    '''
    # Option to reload from previous state
    checkpoint = torch.load(file_name_scheme + '_dict.pth')
    model.load_state_dict(checkpoint)

    with open(file_name_scheme + '_hyperparameters.json', 'r') as file:
        model_hyperparameters = json.load(file)

    print('loaded model learnrate = ', model_hyperparameters['learnrate'])

    return model, model_hyperparameters
