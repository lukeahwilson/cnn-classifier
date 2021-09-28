#!/usr/bin/python
# PROGRAMMER: Luke Wilson
# DATE CREATED: 2021-09-27
# REVISED DATE: 2021-09-28
# PURPOSE:
#   - Provide utility functions for import into main
#       o o1_load_data for loading data for training and classification
#       o o2_map_labels for mapping labels on data against data indexes
#       o o3_process_data for processing data into suitable conditions to be inputed into model
#       o o4_attempt_overfitting for attempting to overfit a subset of the available data as an initial fitness test for the concept model
#       o o5_train_model for training the model on the available training data
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
import json
%matplotlib inline


def o1_get_input_args():
    """
    Creates and stores command line arguments inputted by the user. Attaches default arguments and help text to aid user.
    Command Line Arguments:
        1. Directory as --dir
        2. CNN Model as --model
        3. Data Name Dictionary as --names
    This function returns these arguments as an ArgumentParser object.
    Parameters:
        - None
    Returns:
        - Stored command line arguments as an Argument Parser Object with parse_args() data structure
    """

    parser = argparse.ArgumentParser(description = 'Classify input images and benchmark performance')
    parser.add_argument('--dir', type=str, default='~/Programming Data/Flower_data/', help='input path for data directory')
    parser.add_argument('--model', type=str, default='googlenet', help='select pretrained model', choices=['googlenet', 'alexnet', 'resnet'])
    parser.add_argument('--names', type=str, default='', help='flower_to_name.json')

    return parser.parse_args() #return parsed arguments


def o1_load_processed_data(data_dir):

    #Define data pathway
    train_dir = data_dir + 'train'
    valid_dir = data_dir + 'valid'
    test_dir = data_dir + 'test'
    overfit_dir = data_dir + 'overfit'
    predict_dir = data_dir + 'predict'

    #Create datasets and transform them with defined transforms above
    train_data = datasets.ImageFolder(train_dir, transform=o3_process_data(train))
    valid_data = datasets.ImageFolder(valid_dir, transform=o3_process_data(valid))
    test_data = datasets.ImageFolder(test_dir, transform=o3_process_data(test))
    overfit_data = datasets.ImageFolder(overfit_dir, transform=o3_process_data(overfit))
    predict = datasets.ImageFolder(test_dir, transform=o3_process_data(predict))

    #Turn datasets into generators that can be accessed by an iterator or loop
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=256, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)
    overfit_loader = torch.utils.data.DataLoader(overfit_data, batch_size=8, shuffle=True)
    predict_loader = torch.utils.data.DataLoader(game_data, batch_size=2, shuffle=True)

    with open(data_dir + 'flower_to_name.json', 'r') as f:
        flower_name_dic = json.load(f)

        #Create file pathway for hyperparameter saving to JSON format later
        file_hyperparameters = 'flower-classifier-googlenet-hyperparameters.json'

    return


def o2_process_data(transform_request):
    #Define transforms for training, validation, overfitting, and test sets to convert to desirable tensors for processing
    image_1d_size = 224

    predict_transform = transforms.Compose([transforms.Resize(int(np.round_(256, decimals=0))),
                                            transforms.CenterCrop(image_1d_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    inverse_transform = transforms.Compose([transforms.Normalize([0, 0, 0], [1/0.229, 1/0.224, 1/0.225]),
                                            transforms.Normalize([-0.485, -0.456, -0.406], [1, 1, 1])])

    train_transform = transforms.Compose([transforms.RandomRotation(20),
                                          transforms.RandomResizedCrop(image_1d_size),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    valid_transform = transforms.Compose([transforms.Resize(int(np.round_(image_1d_size*1.1, decimals=0))),
                                          transforms.CenterCrop(image_1d_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([transforms.Resize(int(np.round_(image_1d_size*1.1, decimals=0))),
                                         transforms.CenterCrop(image_1d_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    game_transform = transforms.Compose([transforms.Resize(int(np.round_(image_1d_size*1.1, decimals=0))),
                                         transforms.CenterCrop(image_1d_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    overfit_transform = train_transform

    return = globals()[transform_request + '_transform')


def o3_attempt_overfitting():

def o4_train_model():

print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(torch.cuda.current_device()))
print(torch.cuda.is_available())

def o5_predict_data():


if __name__ == "__main__":
    main()
