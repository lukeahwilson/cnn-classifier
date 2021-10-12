#!/usr/bin/python
# PROGRAMMER: Luke Wilson
# DATE CREATED: 2021-09-27
# REVISED DATE: 2021-09-28
# PURPOSE:
#   - The project is broken down into multiple steps:
#   - Load an image dataset
#   - Preprocess the dataset
#   - Load a pre-trained network
#   - Attach new fully connected layers to network
#   - Train these layers against pretrained features
#   - Save the hyperparameters, training history, and training state
#   - Provide opportunity for retraining or predictions
#   - Make predictions on newly provided data
#   - Use argparse to parse user inputs when calling python script:
#       o python convolutional-classifier.py --## <'string'> --## <'string'> --dir <directory>
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

from cnn_model_functions import *
from cnn_operational_functions  import *


def main():
    """
    1. load data
    2. process data
    3. map data
    4. display data examples
    5. download pretrained model
    6. create a classifier
    7. load model if desired
    8. option for overfitting
    9. train the model
    10. plot training history
    11. test the model
    12. save the model prompt
    13. predict data
    14. show predictions
    """

    # Record start time
    start_program_time = time.time()

    # Get arguments
    arg = o1_get_input_args()


    # Get processed data
    dict_datasets, data_labels_dic = o2_load_processed_data(arg.dir)
    dict_data_loaders = o4_data_iterator(dict_datasets)

    #Create file pathway for hyperparameter saving to JSON format later
    file_name_scheme = os.path.basename(os.path.dirname(arg.dir)) + '_' + arg.model
    criterion = nn.NLLLoss()

    # Download a classifer model for use
    model = m1_create_classifier(arg.model, len(dict_datasets['train_data'].classes))

    if arg.train == 'y':
        print('Displaying an example processed image from the training and validation sets')
        # plt.imshow(dict_datasets['train_data'][0][0].numpy().transpose((1, 2, 0)))
        # plt.imshow(dict_datasets['valid_data'][0][0].numpy().transpose((1, 2, 0)))

        if u1_time_limited_input('Check model can overfit small dataset'):
            overfit_model, model_hyperparamaters = o5_train_model(model, dict_data_loaders, arg.epoch, 'overfit_loader', criterion)

        if u1_time_limited_input('Continue with complete model training'):
            model, model_hyperparamaters = o5_train_model(model, dict_data_loaders, arg.epoch, 'train_loader', criterion)

        if u1_time_limited_input('Would you like to test the model'):
            t1 = time.time()
            test_count_correct, ave_test_loss = o7_model_no_backprop(model, dict_data_loaders['testing_loader'], criterion)
            print('testing Loss: {:.3f}.. '.format(ave_test_loss),
                'testing Accuracy: {:.3f}'.format(test_count_correct / len(dict_data_loaders['testing_loader'].dataset)),
                'Runtime - {:.0f} seconds'.format((time.time() - t1)))

        #Save the model hyperparameters and the locations in which the CNN training activated and deactivated
        if u1_time_limited_input('Would you like to save the model'):
            m2_save_model_checkpoint(model, file_name_scheme, model_hyperparameters)


    if arg.train == 'n':
        model, model_hyperparamaters = m3_load_model_checkpoint(model, file_name_scheme)

        learnrate = model_hyperparameters['learnrate']
        training_loss_history = model_hyperparameters['training_loss_history']
        validate_loss_history = model_hyperparameters['validate_loss_history']
        epoch_on = model_hyperparameters['epoch_on']
        running_count = model_hyperparameters['running_count']

        print('The model is ready to provide predictions')

    #
    #
    #
    # u3_load_model_checkpoint():
    #
    # o3_attempt_overfitting_model():
    #
    # o4_train_model():
    #
    # u4_plot_training_history():
    #
    # u5_test_model():
    #
    # u6_save_model_checkpoint():
    #
    # o5_predict_data():
    #
    # u7_show_prediction():

if __name__ == "__main__":
    main()
