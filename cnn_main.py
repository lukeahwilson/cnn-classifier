#!/usr/bin/python
# PROGRAMMER: Luke Wilson
# DATE CREATED: 2021-09-27
# REVISED DATE: 2021-09-27
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

    o1_load_data():

    o2_map_labels():

    o3_process_data():

    u1_show_data():

    c1_download_pretrained_model():

    c2_create_classifier():

    u2_load_model_checkpoint():

    o4_attempt_overfitting_model():

    o5_train_model():

    u3_plot_training_history():

    u4_test_model():

    u5_save_model_checkpoint():

    o6_predict_data():

    u6_show_prediction():

if __name__ == "__main__":
    main()
