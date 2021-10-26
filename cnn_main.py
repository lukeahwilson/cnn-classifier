#!/usr/bin/python
# PROGRAMMER: Luke Wilson
# DATE CREATED: 2021-09-27
# REVISED DATE: 2021-09-28
# PURPOSE: Train a model and use it to make predictions as required by the user
#   - Retrieve command line arguments to dictate model type, training parameters, and data
#   - Load an image dataset, process the image data, and convert it to a generator
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
from PIL import Image
import matplotlib.pyplot as plt

from cnn_model_functions import *
from cnn_utility_functions import *
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
    arg = u1_get_input_args()

    # Get processed data
    dict_datasets, dict_data_labels, dict_class_labels = u2_load_processed_data(arg.dir)
    dict_data_loaders = u4_data_iterator(dict_datasets)

    #Create file pathway for hyperparameter saving to JSON format later
    file_name_scheme = 'saved-models/' + os.path.basename(os.path.dirname(arg.dir)) + '_' + arg.model + '_' + str(arg.layer) + 'lay'
    criterion = nn.NLLLoss()

    # Download a classifer model for use
    model = m1_create_classifier(arg.model, arg.layer, len(dict_datasets['train_data'].classes))

    if arg.train == 'y':
        print('Displaying an example processed image from the training set..\n')
        plt.imshow(random.choice(dict_datasets['train_data'])[0].numpy().transpose((1, 2, 0)))
        plt.show(block=False)
        plt.pause(2)
        plt.close()

        if u5_time_limited_input('Check model can overfit small dataset?'):
            overfit_model, model_hyperparameters = o1_train_model(model, dict_data_loaders, arg.epoch, arg.learn, 'overfit_loader', criterion)
            o5_plot_training_history(arg.model, model_hyperparameters)
            plt.savefig(file_name_scheme + '_training_history_overfit.png')
            print('Saved overfit training history to project directory')

        if u5_time_limited_input('Continue with complete model training?'):
            model, model_hyperparameters = o1_train_model(model, dict_data_loaders, arg.epoch, arg.learn, 'train_loader', criterion)
            o5_plot_training_history(arg.model, model_hyperparameters)
            plt.savefig(file_name_scheme + '_training_history_complete.png')
            print('Saved complete training history to project directory')
            #Save the model hyperparameters and the locations in which the CNN training activated and deactivated
            if u5_time_limited_input('Would you like to save the model?'):
                m2_save_model_checkpoint(model, file_name_scheme, model_hyperparameters)

    if arg.train == 'n':
        model, model_hyperparameters = m3_load_model_checkpoint(model, file_name_scheme)
        o5_plot_training_history(arg.model, model_hyperparameters)
        plt.show(block=False)
        plt.pause(3)
        plt.close()

        print('The model is ready to provide predictions')


    if u5_time_limited_input('Would you like to test the model?'):
        t1 = time.time()
        test_count_correct, ave_test_loss = o3_model_no_backprop(model, dict_data_loaders['testing_loader'], criterion)
        print('testing Loss: {:.3f}.. '.format(ave_test_loss),
            'testing Accuracy: {:.3f}'.format(test_count_correct / len(dict_data_loaders['testing_loader'].dataset)),
            'Runtime - {:.0f} seconds'.format((time.time() - t1)))

    if u5_time_limited_input('Would you like to use the model for inference?'):
        dict_prediction_results = o6_predict_data(model, dict_data_loaders['predict_loader'], dict_data_labels, dict_class_labels)

    o7_show_prediction(arg.dir, dict_prediction_results)


if __name__ == "__main__":
    main()
