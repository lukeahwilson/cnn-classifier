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

import json

def u3_plot_training_history(training_loss_history, validate_loss_history, ):

    plt.plot(training_loss_history, label='Training Training Loss')
    plt.plot(validate_loss_history, label='Validate Training Loss')
    plt.vlines(
        colors = 'black',
        x = epoch_on,
        ymin = min(training_loss_history),
        ymax = max(training_loss_history[5:]),
        linestyles = 'dotted',
        label = 'CNN Layers Activated'
    ).set_clip_on(False)
    plt.vlines(
        colors = 'black',
        x = epoch_on + running_count,
        ymin = min(training_loss_history),
        ymax = max(training_loss_history[5:]),
        linestyles = 'dotted',
        label = 'CNN Layers Deactivated'
    ).set_clip_on(False)
    plt.ylabel('Total Loss')
    plt.xlabel('Total Epoch ({})'.format(len(training_loss_history)))
    plt.legend(frameon=False)

def u4_test_model():

# for e in range(epochs):
    epoch_test_loss = 0 # initialize total testing loss for this epoch
    test_count_correct = 0 # initialize total correct predictions on test set
    model.eval() # set model to evaluate mode to deactivate generalizing operations such as dropout and leverage full model
    with torch.no_grad(): # turn off gradient tracking and calculation for computational efficiency
        for test_images, test_labels in test_loader: # cycle through testing data to observe performance
            test_images, test_labels = test_images.to(device), test_labels.to(device) # move data to GPU
            log_out = model(test_images) # obtain the logarithmic probability from the model
            loss = criterion(log_out, test_labels) # calculate loss (error) for this image batch based on criterion
            epoch_test_loss += loss.item() # add testing loss to total test loss this epoch, convert to value with .item()

            out = torch.exp(log_out) # obtain probability from the logarithmic probability calculated by the model
            highest_prob, chosen_class = out.topk(1, dim=1) # obtain the chosen classes based on greatest calculated probability
            equals = chosen_class.view(test_labels.shape) == test_labels # determine how many correct matches were made in this batch
            test_count_correct += equals.sum()  # add the count of correct matches this batch to the total running this epoch

        ave_testing_loss = epoch_test_loss / len(test_loader) # determine average loss per batch of testing images

    print('Epoch: {}/{}.. '.format(e+1, epochs),
        'testing Loss: {:.3f}.. '.format(ave_testing_loss),
        'testing Accuracy: {:.3f}'.format(test_count_correct / len(test_loader.dataset)),
        'Runtime - {:.0f} seconds'.format((time.time() - t0)))

# def get_image(image_path):
    ''' Process raw image for input to deep learning model
    '''
    image_open = Image.open(image_path) # access image at pathway, open the image and store it as a PIL image
    tensor_image = flower_transform(image_open) # transform PIL image and simultaneously convert image to a tensor (no need for .clone().detach())
    input_image = torch.unsqueeze(tensor_image, 0) # change image shape from a stand alone image tensor, to a list of image tensors with length = 1
    return input_image # return processed image

# def prediction(image_path, model, topk=5):
    ''' Compute probabilities for various classes for an image using a trained deep learning model.
    '''
    model.eval()
    with torch.no_grad():
        input_image = get_image(image_path)
        prediction = torch.exp(model(input_image))
        probabilities, classes = prediction.topk(topk)
    return probabilities, classes


def u5_show_prediction():
