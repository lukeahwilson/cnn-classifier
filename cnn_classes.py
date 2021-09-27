#!/usr/bin/python
# PROGRAMMER: Luke Wilson
# DATE CREATED: 2021-09-27
# REVISED DATE: 2021-09-27
# PURPOSE:
#   - Provide utility functions for import into main
#       o c1_download_pretrained_model for downloading desireable pretrained models to use
#       o c2_create_classifier() for reassigning output layers to newly attached layers and creating initialized classifier
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




def c1_download_pretrained_model():


def c2_create_classifier():
