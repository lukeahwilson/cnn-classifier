#!/usr/bin/python
# PROGRAMMER: Luke Wilson
# DATE CREATED: 2021-09-27
# REVISED DATE: 2021-09-28
# PURPOSE:
#   - Provide utility functions for import into main
#       o u1_show_data for displaying the loaded and processed data
#       o u2_load_model_checkpoint for loading previously saved model training state and hyperparameters
#       o u3_plot_training_history for plotting the training history and performance
#       o u4_test_model for testing the model on training data to benchmark performance
#       o u6_show_prediction for displaying the predicted image with the prediction
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



def u1_show_data():


def u2_map_labels():
    import json

    with open('flower_to_name.json', 'r') as f:
        flower_name_dic = json.load(f)

    #Create file pathway for hyperparameter saving to JSON format later
    file_hyperparameters = 'flower-classifier-googlenet-hyperparameters.json'

    # this single line of code would have saved me an incredibly large amount of problem solving haha, oh well!
    # reversed_dictionary = {value : key for (key, value) in a_dictionary.items()}


def u3_plot_training_history():


def u4_test_model():



def u5_show_prediction():
