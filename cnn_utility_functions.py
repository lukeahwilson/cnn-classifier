#!/usr/bin/python
# PROGRAMMER: Luke Wilson
# DATE CREATED: 2021-09-27
# REVISED DATE: 2021-09-28
# PURPOSE: Provide utility functions for import into main
#   - o1_load_data for loading data for training and classification
#   - o2_map_labels for mapping labels on data against data indexes
#   - o3_process_data for processing data into suitable conditions to be inputed into model
#   - o4_attempt_overfitting for attempting to overfit a subset of the available data as an initial fitness test for the concept model
#   - o5_train_model for training the model on the available training data
##

# Import required libraries
import json
import argparse #import python argparse function
import os, random
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets, models
from torch import nn, optim
from PIL import Image
from threading import Thread


def u1_get_input_args():
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
    parser.add_argument('--dir', type=str, default= os.path.expanduser('~')+'/Programming Data/Flower_data/', help='input path for data directory')
    parser.add_argument('--train', type=str, default='n', help='yes \'y\' or no \'n\' to retrain this model', choices=['y','n'])
    parser.add_argument('--epoch', type=int, default=100, help='provide the number of epochs for training (default 100)')
    parser.add_argument('--layer', type=int, default=2, help='provide the number of hidden layers to use (default 2)')
    parser.add_argument('--learn', type=int, default=0.003, help='provide the learning rate to begin training (default 0.003)')
    parser.add_argument('--label', type=str, default='', help='flower_to_name.json')
    parser.add_argument('--model', type=str, default='googlenet', help='select pretrained model',
                        choices=['vgg', 'alexnet', 'googlenet', 'densenet', 'resnext', 'shufflenet'])

    return parser.parse_args() #return parsed arguments


def u2_load_processed_data(data_dir):
    """
    Receives a pathway to a folder containing training, .
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

    dict_datasets = {}
    for folder in os.listdir(data_dir):
        if os.path.splitext(folder)[1] == '' and folder != 'predict':
            dict_datasets[folder + '_data'] = datasets.ImageFolder(data_dir + folder, transform=u3_process_data(folder))
        if os.path.splitext(folder)[1] == '' and folder == 'predict':
            predict_transform = u3_process_data(folder)
            dict_datasets['predict_data'] = [(predict_transform(Image.open(data_dir + folder + '/' + filename)), filename) for filename in os.listdir(data_dir + folder)]
        if os.path.splitext(folder)[1] == '.json':
            with open(data_dir + folder, 'r') as f:
                dict_data_labels = json.load(f)
    dict_class_labels = {value : key for (key, value) in dict_datasets['train_data'].class_to_idx.items()}
    return dict_datasets, dict_data_labels, dict_class_labels


def u3_process_data(transform_request):
    #Define transforms for training, validation, overfitting, and test sets to convert to desirable tensors for processing

    #Define image size
    image_1d_size = 224

    predict_transform = transforms.Compose([transforms.Resize(int(np.round_(image_1d_size*1.1, decimals=0))),
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

    return locals()[transform_request + '_transform']


def u4_data_iterator(dict_datasets):
    '''
    # bug, requires every folder to run correctly. Consider removing this function and nesting it directly into training function!
    '''
    dict_data_loaders = {}
    dict_data_loaders['train_loader'] = torch.utils.data.DataLoader(dict_datasets['train_data'], batch_size=128, shuffle=True)
    dict_data_loaders['valid_loader'] = torch.utils.data.DataLoader(dict_datasets['valid_data'], batch_size=64, shuffle=True)
    dict_data_loaders['testing_loader'] = torch.utils.data.DataLoader(dict_datasets['test_data'], batch_size=32, shuffle=True)
    dict_data_loaders['overfit_loader'] = torch.utils.data.DataLoader(dict_datasets['overfit_data'], batch_size=8, shuffle=True)
    dict_data_loaders['predict_loader'] = torch.utils.data.DataLoader(dict_datasets['predict_data'], batch_size=2, shuffle=False)

    return dict_data_loaders


def u5_time_limited_input(prompt):
    TIMEOUT = 10
    prompt = prompt + f': \'y\' for yes, \'n\' for no ({TIMEOUT} seconds to choose): '
    user_input_thread = Thread(target=u6_user_input_prompt, args=(prompt,), daemon = True)
    user_input_thread.start()
    user_input_thread.join(TIMEOUT)
    if not answered:
        print('\n No valid input, proceeding with operation...')
    return choice


def u6_user_input_prompt(prompt, default=True):
    global choice, answered
    choice = default
    answered = False
    while not answered:
        choice = input(prompt)
        if choice == 'Y' or choice == 'y':
            print('User input = Yes\n')
            choice = True
            answered = True
        elif choice == 'N' or choice == 'n':
            choice = False
            answered = True
            print('User input = No\n')
        else:
            choice=choice
            print('Error, please use the character inputs \'Y\' and \'N\'')
