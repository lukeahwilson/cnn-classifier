#!/usr/bin/python
# PROGRAMMER: Luke Wilson
# DATE CREATED: 2021-10-26
# REVISED DATE: 2021-10-26
# PURPOSE: Short script to run the main CNN script iterating through the model choices
#   - Use subprocess routine to provide commands in terminal to run desired script
##

import subprocess

for model in ['vgg', 'alexnet', 'googlenet', 'densenet', 'resnext', 'shufflenet']:
    subprocess.run(['python', 'cnn_main.py', '--train', 'y', '--dir', '/home/workspace/ImageClassifier/Flower_data/', '--model', model])
    print('Completed: ', model)
