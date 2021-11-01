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

# Import libraries
import json
import time, os, random
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets, models
from torch import nn, optim
from PIL import Image
import matplotlib.pyplot as plt


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
    # Initialize attributes, requiring input arguments for the number of hidden layers, and input and output features
    # Use super() for multiple inheritance from nn.Module, use arguments to create in, out, hidden layer attributes
    def __init__(self, in_features, hidden_layers, out_features):
        super().__init__()
        self.in_features = in_features
        self.hidden_layers = hidden_layers
        self.out_features = out_features
        self._index = 1

        # Iterate to create the requested number of hidden layers, tapering down shape by a factor of 2 between layers
        # Setattr is used to create a layer attribute with the fc('index') name and the factored shape
        # Use the required number of out features for the output of the last layer, set dropout to a desired value
        while self._index < self.hidden_layers:
            setattr(self, 'fc'+str(self._index), nn.Linear(round(self.in_features/(2**(self._index-1))),
                            round(self.in_features/(2**self._index))))
            self._index += 1
        setattr(self, 'fc'+str(self._index), nn.Linear(round(self.in_features/(2**(self._index-1))), self.out_features))
        self.dropout = nn.Dropout(p=0.3)

    # Define the forward function that will take an input and compute it through the number of layers
    # Start by flattening the data, then use the number of hidden layers to iterate through each existing layer
    # Use the Relu activation function between layers, apply the defined dropout rate, and return the softmax probability
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
        - Return an integrated CNN architecture by:
            o downloading a pretrained model
            o attaching a fully connected network
        - Leverages the requested pretrained model to provide base features
    Parameters:
        - model_name = base pretrained model
        - hidden_layers = number of hidden layers in final fully connected network
        - out_features = number of classes in data
    Returns:
        - model
    '''
    # Store a dictionary of available models as names to avoid downloading models until a choice has been made
    model_name_dic = {'vgg': 'vgg16', 'alexnet': 'alexnet', 'googlenet': 'googlenet', 'densenet': 'densenet161',
                      'resnext': 'resnext50_32x4d', 'shufflenet': 'shufflenet_v2_x1_0'}

    # Download the pretrained convolutional neural network architecture requested by the user and freeze the parameters
    model = getattr(models, model_name_dic[model_name])(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    # Search the pretrained architecture for the first fully-connected layer and return the number of in features
    for module in list(model.modules()):
        if module._get_name() == 'Linear':
            in_features = module.weight.shape[1]
            break

    # Use the known number of in and out features to ensure compatibility for the attached fully connected layers
    # Replace the fully connected layer(s) at the end of the model with our own fully connected classifier
    setattr(model, list(model._modules.items())[-1][0], Classifier(in_features, hidden_layers, classes_length))

    # Print the name of the model and the architecture of the attached layers, then return the model
    print('\nUsing ', model_name, ' with the following attached ', hidden_layers,
                    ' layer classifier:\n', list(model.children())[-1])
    return model


def m2_save_model_checkpoint(model, file_name_scheme, model_hyperparameters):
    '''
    Purpose:
        - Receive a model, a naming convention, and model hyperparameter
        - Save model checkpoint and hyperparameters
    Parameters:
        - model = model to be saved
        - file_name_scheme = directory and naming convention for saving
        - model_hyperparameters = information about state of model
    Returns:
        - none
    '''
    # Save the model state_dict per the naming convention as a pth file
    torch.save(model.state_dict(), file_name_scheme + '_dict.pth')

    # Save the model hyperparameters per the naming convention as a JSON file
    with open(file_name_scheme + '_hyperparameters.json', 'w') as file:
        json.dump(model_hyperparameters, file)


def m3_load_model_checkpoint(model, file_name_scheme):
    '''
    Purpose:
        - Receive a model, a naming convention, and model hyperparameters
        - Load model checkpoint and hyperparameters
    Parameters:
        - model = model to be loaded
        - file_name_scheme = directory and naming convention for loading
    Returns:
        - model
        - model hyperparameters
    '''
    # Load the model state_dict by using the naming convention to find the file
    checkpoint = torch.load(file_name_scheme + '_dict.pth')
    model.load_state_dict(checkpoint)

    # Load the model hyperparameters by using the naming convention and display the learnrate and train time
    with open(file_name_scheme + '_hyperparameters.json', 'r') as file:
        model_hyperparameters = json.load(file)
    print('\nLoaded model learnrate = {:.2e}..'.format( model_hyperparameters['learnrate']),
          'Loaded model training time = {:.0f} min\n'.format( model_hyperparameters['training_time']))
    return model, model_hyperparameters
