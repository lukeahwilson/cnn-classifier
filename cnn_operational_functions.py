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
    parser.add_argument('--dir', type=str, default= os.path.expanduser('~')+'/Programming Data/Flower_data/', help='input path for data directory')
    parser.add_argument('--model', type=str, default='googlenet', help='select pretrained model', choices=['googlenet', 'alexnet', 'resnet'])
    parser.add_argument('--train', type=str, default='n', help='yes 'y' or no 'n' to retrain this model', choices=['y','n'])
    parser.add_argument('--names', type=str, default='', help='flower_to_name.json')
    return parser.parse_args() #return parsed arguments


def o1_load_processed_data(data_dir):
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
    #         dict_datadir[folder] =
            dict_datasets[folder + '_data'] = datasets.ImageFolder(data_dir + folder, transform=o2_process_data(folder))
        if os.path.splitext(folder)[1] == '.json':
            with open(data_dir + folder, 'r') as f:
                data_name_dic = json.load(f)
    return dict_datasets


    #Create file pathway for hyperparameter saving to JSON format later
    file_hyperparameters = 'flower-classifier-googlenet-hyperparameters.json'

    # this single line of code would have saved me an incredibly large amount of problem solving haha, oh well!
    # reversed_dictionary = {value : key for (key, value) in a_dictionary.items()}


def o1_data_to_iterator(dict_datasets):
    train_loader = torch.utils.data.DataLoader(dict_datasets['train_data'], batch_size=256, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dict_datasets['valid_data'], batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dict_datasets['test_data'], batch_size=32, shuffle=True)
    overfit_loader = torch.utils.data.DataLoader(dict_datasets['overfit_data'], batch_size=8, shuffle=True)

    #Create file pathway for hyperparameter saving to JSON format later
    file_hyperparameters = 'flower-classifier-googlenet-hyperparameters.json'
    return loader


def o2_process_data(transform_request):
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

    return locals()[transform_request + '_transform')


def o3_train_model():
    # Check model can overfit the data when using a miniscule sample size, looking for high accuracy on a few images
    print("Using GPU" if torch.cuda.is_available() else "WARNING")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    model.to(device)


    t0 = time.time() # initialize start time for running training

    running_count = 0 # initialize running count in order to track number of epochs fine tuning deeper network
    running = False # initialize running variable to start system with deeper network frozen

    decayover = 0.9 # hyperparameter decay factor for decaying learning rate
    epochsover = 200 # hyperparameter number of epochs

    decay = 0.6 # hyperparameter decay factor for decaying learning rate
    epochs = 60 # hyperparameter number of epochs

    for e in range(epochs):
        epoch_train_loss = 0 # initialize total training loss for this epoch
        model.train() # Set model to training mode to activate operations such as dropout
        for train_images, train_labels in train_loader: # cycle through training data
            train_images, train_labels = train_images.to(device), train_labels.to(device) # move data to GPU

            optimizer.zero_grad() # clear gradient history
            log_out = model(train_images) # run image through model to get logarithmic probability
            loss = criterion(log_out, train_labels) # calculate loss (error) for this image batch based on criterion

            loss.backward() # backpropogate gradients through model based on error
            optimizer.step() # update weights in model based on calculated gradient information
            epoch_train_loss += loss.item() # add training loss to total train loss this epoch, convert to value with .item()

        else:
            epoch_valid_loss = 0 # initialize total validate loss for this epoch
            valid_count_correct = 0 # initialize total correct predictions on valid set
            model.eval() # set model to evaluate mode to deactivate generalizing operations such as dropout and leverage full model
            with torch.no_grad(): # turn off gradient tracking and calculation for computational efficiency
                for valid_images, valid_labels in valid_loader: # cycle through validate data to observe performance
                    valid_images, valid_labels = valid_images.to(device), valid_labels.to(device) # move data to GPU
                    log_out = model(valid_images) # obtain the logarithmic probability from the model
                    loss = criterion(log_out, valid_labels) # calculate loss (error) for this image batch based on criterion
                    epoch_valid_loss += loss.item() # add validate loss to total valid loss this epoch, convert to value with .item()

                    out = torch.exp(log_out) # obtain probability from the logarithmic probability calculated by the model
                    highest_prob, chosen_class = out.topk(1, dim=1) # obtain the chosen classes based on greavalid calculated probability
                    equals = chosen_class.view(valid_labels.shape) == valid_labels # determine how many correct matches were made in this batch
                    valid_count_correct += equals.sum()  # add the count of correct matches this batch to the total running this epoch

            ave_training_loss = epoch_train_loss / len(train_loader) # determine average loss per batch of training images
            ave_validate_loss = epoch_valid_loss / len(valid_loader) # determine average loss per batch of validate images
            training_loss_history.append(ave_training_loss) # append ave training loss to history of training losses
            validate_loss_history.append(ave_validate_loss) # append ave validate loss to history of validate losses

            print('Epoch: {}/{}.. '.format(e+1, epochs),
                'Training Loss: {:.3f}.. '.format(ave_training_loss),
                'Validate Loss: {:.3f}.. '.format(ave_validate_loss),
                'Validate Accuracy: {:.3f}'.format(valid_count_correct / len(valid_loader.dataset)),
                'Runtime - {:.0f} minutes'.format((time.time() - t0)/60))

            if len(training_loss_history) > 3: # hold loop until training_loss_history has enough elements to satisfy search requirements
                if -3*learnrate*decay*decay*training_loss_history[0] > np.mean([training_loss_history[-2]-training_loss_history[-1], training_loss_history[-3]-training_loss_history[-2]]):
                    # if the average of the last 2 training loss slopes is less than the original loss factored down by the learnrate, the decay, and a factor of 3, then decay the learnrate
                    learnrate *= decay # multiply learnrate by the decay hyperparamater
                    optimizer = optim.Adam(model.fc.parameters(), lr=learnrate, weight_decay=weightdecay) # revise the optimizer to use the new learnrate
                    print('\nLearnrate changed to: {:f}\n'.format(learnrate))
                if learnrate <= startlearn*decay**(9*(decay**3)) and running_count == 0: # super messy, I wanted a general expression that chose when to activate the deeper network and this worked
                    for param in model.inception5a.parameters(): # activate parameters in layer 5a once the learning rate has decayed (9*(decay**3)) number of times
                        param.requires_grad = True
                    for param in model.inception5b.parameters(): # activate parameters in layer 5b once the learning rate has decayed (9*(decay**3)) number of times
                        param.requires_grad = True
                    print('\nConvolutional Layers I5A and I5B Training Activated\n')
                    epoch_on = e
                    running = True # change the running parameter to True so that the future loop can start counting epochs that have run
                if running: # if running, add to count for the number of epochs run
                    running_count +=1
                if running and running_count > epochs/5: # deactivate parameters if running, add the count has reached its limiter
                    for param in model.inception5a.parameters():
                        param.requires_grad = False
                    for param in model.inception5b.parameters():
                        param.requires_grad = False
                    print('\nConvolutional Layers I5A and I5B Training Deactivated\n')
                    running = False
            if np.mean([overfit_loss_history[-1], overfit_loss_history[-2], overfit_loss_history[-3]]) < 0.0001:
                print('\nModel successfully overfit images')
                break

        if e+1 == epochs:
            print('\nModel failed to overfit images')


def o4_train_model():
    # Check model can overfit the data when using a miniscule sample size, looking for high accuracy on a few images
    print("Using GPU" if torch.cuda.is_available() else "WARNING")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    model.to(device)



def o5_predict_data():


if __name__ == "__main__":
    main()
