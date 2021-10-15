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
    parser.add_argument('--epoch', type=str, default=100, help='provide the number of epochs for training (default 100)')
    parser.add_argument('--label', type=str, default='', help='flower_to_name.json')
    parser.add_argument('--model', type=str, default='googlenet', help='select pretrained model', choices=['vgg', 'alexnet', 'googlenet',
                        'densenet', 'inception', 'resnext', 'shufflenet'])

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
        if os.path.splitext(folder)[1] == '.json':
            with open(data_dir + folder, 'r') as f:
                data_labels_dic = json.load(f)
                data_labels_dic = {value : key for (key, value) in data_labels_dic.items()}
    return dict_datasets, data_labels_dic


def u3_process_data(transform_request):
    #Define transforms for training, validation, overfitting, and test sets to convert to desirable tensors for processing

    #Define image size
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


def u6_user_input_prompt(prompt):
    global choice, answered
    choice = True
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
            choice = False
            print('Error, please use the character inputs \'Y\' and \'N\'')


# def u7_get_image(image_path):
#     ''' Process raw image for input to deep learning model
#     '''
#     image_open = Image.open(image_path) # access image at pathway, open the image and store it as a PIL image
#     tensor_image = flower_transform(image_open) # transform PIL image and simultaneously convert image to a tensor (no need for .clone().detach())
#     input_image = torch.unsqueeze(tensor_image, 0) # change image shape from a stand alone image tensor, to a list of image tensors with length = 1
#     return input_image # return processed image


# def u8_show_image():
