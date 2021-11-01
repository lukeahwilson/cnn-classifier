#!/usr/bin/python
# PROGRAMMER: Luke Wilson
# DATE CREATED: 2021-10-12
# REVISED DATE: 2021-01-01
# PURPOSE: Provide utility functions for import into main
#   - u1_get_input_args()
#   - u2_load_processed_data(data_dir)
#   - u3_process_data(transform_request)
#   - u4_data_iterator(dict_datasets)
#   - u5_time_limited_input(prompt, default=True)
#   - u6_user_input_prompt(prompt, default)
##

# Import libraries
import json
import argparse
import os, random
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets, models
from torch import nn, optim
from PIL import Image
from threading import Thread


def u1_get_input_args():
    '''
    Purpose:
        - Creates and stores command line arguments inputted by the user.
        - Attaches default arguments and help text to aid user.
    Command Line Arguments:
        1. Data directory as --dir
        2. Choose to load model as --load
        3. Choose to train model as --train
        4. Define number of training epochs as --epoch
        5. Define network number of hidden layers --layer
        6. Define learnrate as --learn
        7. Choose pretrained CNN model as --model
    Returns:
        - Stored command line arguments as an Argument Parser Object with parse_args() data structure
    '''
    parser = argparse.ArgumentParser(description = 'Classify input images and benchmark performance')
    parser.add_argument('--dir', type=str, default= 'Flower_data', help='input path for data directory')
    parser.add_argument('--load', type=str, default='n', help='yes \'y\' or no \'n\' to load state_dict for model', choices=['y','n'])
    parser.add_argument('--train', type=str, default='n', help='yes \'y\' or no \'n\' to retrain this model', choices=['y','n'])
    parser.add_argument('--epoch', type=int, default=50, help='provide the number of epochs for training (default 100)')
    parser.add_argument('--layer', type=int, default=2, help='provide the number of hidden layers to use (default 2)')
    parser.add_argument('--learn', type=int, default=0.003, help='provide the learning rate to begin training (default 0.003)')
    parser.add_argument('--model', type=str, default='googlenet', help='select pretrained model',
                        choices=['vgg', 'alexnet', 'googlenet', 'densenet', 'resnext', 'shufflenet'])
    return parser.parse_args()


def u2_load_processed_data(data_dir):
    '''
    Purpose:
        - Access data directory and produce a dictionary of datasets
        - Create a dictionary of the class labels and read in the data labels
    Parameters:
        - data_dir = pathway to the data
    Returns:
        - dictionary of datasets
        - dictionary of data labels
        - dictionary of class labels
    '''
    # Initialize empty dictionaries to hold data and data labels
    dict_datasets = {}
    dict_data_labels = {}

    # Iterate through folders in the data directory
    for folder in os.listdir(data_dir):

        # If data exists, create datasets for overfitting, testing, training, and validating data
        if folder in ['overfit', 'test', 'train', 'valid']:
            dict_datasets[folder + '_data'] = datasets.ImageFolder(data_dir + folder, transform=u3_process_data(folder))

        # If data for inference exists, create a dataset from the predict folder
        if folder == 'predict':
            predict_transform = u3_process_data(folder)
            dict_datasets['predict_data'] = [(predict_transform(Image.open(data_dir + folder + '/' + filename)),
                            filename) for filename in os.listdir(data_dir + folder)]

        # If a data names are added to the data directory as a json, open it and read into data label dictionary
        if os.path.splitext(folder)[1] == '.json':
            with open(data_dir + folder, 'r') as f:
                dict_data_labels = json.load(f)

    # Create a dictionary connecting class indexes to class labels, return the datasets and label dictionaries
    dict_class_labels = {value : key for (key, value) in dict_datasets['train_data'].class_to_idx.items()}
    return dict_datasets, dict_data_labels, dict_class_labels


def u3_process_data(transform_request):
    '''
    Purpose:
        - Define an assortment of transforms for application to specific datasets
        - Return the appropriate transformation that corresponds to the inputted request
        - Defined transforms are composed of a sequence of individual transform operations
        - Depending on the needs of each data set, a transform will use specific operations
    Parameters:
        - transformation_request = selected transformation type
    Returns:
        - transform that corresponds to the request
    '''
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
    Purpose:
        - Receive a dictionary of datasets
        - Convert each dataset to a dataLoader
        - Return a dictionary of dataloaders
    Parameters:
        - dict_datasets = dictionary of datasets
    Returns:
        - dict_data_loaders = dictionary of dataloaders
    '''
    dict_data_loaders = {}
    for dataset in dict_datasets:
        loader_type = dataset.split('_')[0] + '_loader'
        dict_data_loaders[loader_type] = torch.utils.data.DataLoader(dict_datasets[dataset], batch_size=128, shuffle=True)
    return dict_data_loaders


def u5_time_limited_input(prompt, default=True):
    '''
    Purpose:
        - Receive text and start a thread to initiate a user input prompt with that text
        - Track thread time and limit time to an established TIMEOUT limit
        - Return user input or after the TIMEOUT limit is reached return the default choice
    Parameters:
        - prompt = specific question text for display
        - default = default choice if no user input is provided
    Returns:
        - choice = the user input or the default
    '''
    TIMEOUT = 10
    prompt = prompt + f': \'y\' for yes, \'n\' for no ({TIMEOUT} seconds to choose): '
    user_input_thread = Thread(target=u6_user_input_prompt, args=(prompt, default), daemon = True)
    user_input_thread.start() # Start the thread, calling the user input function
    user_input_thread.join(TIMEOUT) # Limit the thread to the TIMEOUT time limit
    if not answered:
        print('\n No valid input, proceeding with operation...\n')
    return choice


def u6_user_input_prompt(prompt, default):
    '''
    Purpose:
        - Receive a prompt and use it for user input prompting
        - Once answered return True or False if input is yes or no
        - Ask question again if the input is incorrect
    Parameters:
        - prompt = complete user input question text for display
        - default = default choice if no user input is provided
    Returns:
        - choice = the user input or the default
    '''
    global choice, answered # Global variables are required to communicate input statuses back to the thread manager
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
