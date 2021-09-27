#!/usr/bin/python
# PROGRAMMER: Luke Wilson
# DATE CREATED: 2021-09-27
# REVISED DATE: 2021-09-27
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
%matplotlib inline



def o1_load_data():

def o2_map_labels():

def o3_process_data():

def o4_attempt_overfitting():

def o5_train_model():

print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(torch.cuda.current_device()))
print(torch.cuda.is_available())

def o6_predict_data():


if __name__ == "__main__":
    main()
