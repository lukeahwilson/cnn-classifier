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
import matplotlib.pyplot as plt
import time
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets, models
from torch import nn, optim
from PIL import Image


def o1_train_model(model, dict_data_loaders, epoch, type_loader, criterion):

    print("Using GPU" if torch.cuda.is_available() else "WARNING")
    t0 = time.time() # initialize start time for running training

    running_count = 0 # initialize running count in order to track number of epochs fine tuning deeper network
    running = False # initialize running variable to start system with deeper network frozen

    # Define default hyperparameters: learning rate and weight decay
    model_hyperparameters = {'learnrate': 0.003,
                         'training_loss_history': [],
                         'validate_loss_history': [],
                         'epoch_on': [],
                         'running_count': 0,
                         'weightdecay' : 0.00001}

    startlearn = model_hyperparameters['learnrate']

    # Only train the classifier (new_output) parameters, feature parameters are frozen
    optimizer = optim.Adam(model.new_output.parameters(), lr=model_hyperparameters['learnrate'], weight_decay=model_hyperparameters['weightdecay'])

    if type_loader == 'overfit_loader':
        decay = 0.9 # hyperparameter decay factor for decaying learning rate
        epoch = 200 # hyperparameter number of epochs

    if type_loader == 'train_loader':
        decay = 0.6 # hyperparameter decay factor for decaying learning rate

    for e in range(epoch):

        model, ave_training_loss = o2_model_backprop(model, dict_data_loaders[type_loader], optimizer, criterion)
        epoch_count_correct, ave_validate_loss = o3_model_no_backprop(model, dict_data_loaders['valid_loader'], criterion)

        model_hyperparameters['training_loss_history'].append(ave_training_loss) # append ave training loss to history of training losses
        model_hyperparameters['validate_loss_history'].append(ave_validate_loss) # append ave validate loss to history of validate losses

        print('Epoch: {}/{}.. '.format(e+1, epoch),
            'Train Loss: {:.3f}.. '.format(ave_training_loss),
            'Valid Loss: {:.3f}.. '.format(ave_validate_loss),
            'Valid Accuracy: {:.3f}.. '.format(epoch_count_correct / len(dict_data_loaders['valid_loader'].dataset)),
            'Runtime - {:.0f} mins'.format((time.time() - t0)/60))

        training_loss_history = model_hyperparameters['training_loss_history']
        if len(training_loss_history) > 3: # hold loop until training_loss_history has enough elements to satisfy search requirements
            if -3*model_hyperparameters['learnrate']*decay*decay*training_loss_history[0] > np.mean([training_loss_history[-2]-training_loss_history[-1], training_loss_history[-3]-training_loss_history[-2]]):
                # if the average of the last 2 training loss slopes is less than the original loss factored down by the learnrate, the decay, and a factor of 3, then decay the learnrate
                model_hyperparameters['learnrate'] *= decay # multiply learnrate by the decay hyperparameter
                optimizer = optim.Adam(model.new_output.parameters(), lr=model_hyperparameters['learnrate'], weight_decay=model_hyperparameters['weightdecay']) # revise the optimizer to use the new learnrate
                print('Learnrate changed to: {:f}'.format(model_hyperparameters['learnrate']))
            if model_hyperparameters['learnrate'] <= startlearn*decay**(9*(decay**3)) and model_hyperparameters['running_count'] == 0: # super messy, I wanted a general expression that chose when to activate the deeper network and this worked
                model = o4_control_model_grad(model, True)
                model_hyperparameters['epoch_on'] = e
                running = True # change the running parameter to True so that the future loop can start counting epochs that have run
            if running: # if running, add to count for the number of epochs run
                model_hyperparameters['running_count'] +=1
            if running and model_hyperparameters['running_count'] > epoch/5: # deactivate parameters if running, add the count has reached its limiter
                model = o4_control_model_grad(model, False)
                running = False
            if type_loader == 'overfit_loader':
                if np.mean([training_loss_history[-1], training_loss_history[-2], training_loss_history[-3]]) < 0.0002:
                    print('\nModel successfully overfit images\n')
                    return model, model_hyperparameters
                if e+1 == epoch:
                    print('\nModel failed to overfit images\n')

    return model, model_hyperparameters


def o2_model_backprop(model, data_loader, optimizer, criterion):
    # Check model can overfit the data when using a miniscule sample size, looking for high accuracy on a few images
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    model.to(device)

    epoch_train_loss = 0 # initialize total training loss for this epoch
    model.train() # Set model to training mode to activate operations such as dropout
    for images, labels in data_loader: # cycle through training data
        images, labels = images.to(device), labels.to(device) # move data to GPU

        optimizer.zero_grad() # clear gradient history
        log_out = model(images) # run image through model to get logarithmic probability
        loss = criterion(log_out, labels) # calculate loss (error) for this image batch based on criterion

        loss.backward() # backpropogate gradients through model based on error
        optimizer.step() # update weights in model based on calculated gradient information
        epoch_train_loss += loss.item() # add training loss to total train loss this epoch, convert to value with .item()
    ave_training_loss = epoch_train_loss / len(data_loader) # determine average loss per batch of training images

    return model, ave_training_loss


def o3_model_no_backprop(model, data_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    epoch_valid_loss = 0 # initialize total validate loss for this epoch
    epoch_count_correct = 0 # initialize total correct predictions on valid set
    model.eval() # set model to evaluate mode to deactivate generalizing operations such as dropout and leverage full model
    with torch.no_grad(): # turn off gradient tracking and calculation for computational efficiency
        for images, labels in data_loader: # cycle through validate data to observe performance
            images, labels = images.to(device), labels.to(device) # move data to GPU

            log_out = model(images) # obtain the logarithmic probability from the model
            loss = criterion(log_out, labels) # calculate loss (error) for this image batch based on criterion
            epoch_valid_loss += loss.item() # add validate loss to total valid loss this epoch, convert to value with .item()

            out = torch.exp(log_out) # obtain probability from the logarithmic probability calculated by the model
            highest_prob, chosen_class = out.topk(1, dim=1) # obtain the chosen classes based on greavalid calculated probability
            equals = chosen_class.view(labels.shape) == labels # determine how many correct matches were made in this batch
            epoch_count_correct += equals.sum()  # add the count of correct matches this batch to the total running this epoch

        ave_validate_loss = epoch_valid_loss / len(data_loader) # determine average loss per batch of validate images

    return epoch_count_correct, ave_validate_loss


def o4_control_model_grad(model, control=False):
    network_depth = len(list(model.children()))
    param_freeze_depth = network_depth // 2
    controlled_layers = []
    layer_depth = 0

    for layer in model.children():
        layer_depth += 1
        if (network_depth - param_freeze_depth) < layer_depth < network_depth:
            controlled_layers.append(layer._get_name())
            for param in layer.parameters():
                param.requires_grad = control
    print(f'\n Toggle requires_grad = {control}: ', controlled_layers, '\n')
    return model


def o5_plot_training_history(model_name, model_hyperparameters):
    plt.clf()
    plt.plot(model_hyperparameters['training_loss_history'], label='Training Training Loss')
    plt.plot(model_hyperparameters['validate_loss_history'], label='Validate Training Loss')
    if model_hyperparameters['epoch_on']:
        plt.vlines(
            colors = 'black',
            x = model_hyperparameters['epoch_on'],
            ymin = min(model_hyperparameters['training_loss_history']),
            ymax = max(model_hyperparameters['training_loss_history'][3:]),
            linestyles = 'dotted',
            label = 'Deep Layers Activated'
        ).set_clip_on(False)
        plt.vlines(
            colors = 'black',
            x = (model_hyperparameters['epoch_on'] + model_hyperparameters['running_count']),
            ymin = min(model_hyperparameters['training_loss_history']),
            ymax = max(model_hyperparameters['training_loss_history'][3:]),
            linestyles = 'dotted',
            label = 'Deep Layers Deactivated'
        ).set_clip_on(False)
    plt.title(model_name)
    plt.ylabel('Total Loss')
    plt.xlabel('Total Epoch ({})'.format(len(model_hyperparameters['training_loss_history'])))
    plt.legend(frameon=False)

#
# def o6_predict_data(image_path, model, topk=5):
#     ''' Compute probabilities for various classes for an image using a trained deep learning model.
#     '''
#     model.eval()
#     with torch.no_grad():
#         input_image = get_image(image_path)
#         prediction = torch.exp(model(input_image))
#         probabilities, classes = prediction.topk(topk)
#     return probabilities, classes
