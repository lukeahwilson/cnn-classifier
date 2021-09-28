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
%matplotlib inline

from cnn-classes import *
from cnn-utility-functions import *
from cnn-operational-functions  import *


def main():
    start_time = time()
    arg = get_input_arguments()

    o1_load_data(data_dir):

    u1_map_labels():

    o2_process_data():

    u2_show_data():

    c1_download_pretrained_model():

    c2_create_classifier():

    u3_load_model_checkpoint():

    o3_attempt_overfitting_model():

    o4_train_model():

    u4_plot_training_history():

    u5_test_model():

    u6_save_model_checkpoint():

    o5_predict_data():

    u7_show_prediction():

if __name__ == "__main__":
    main()
