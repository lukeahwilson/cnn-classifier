#!/usr/bin/python
# PROGRAMMER: Luke Wilson
# DATE CREATED: 2021-09-27
# REVISED DATE: 2021-10-27
# PURPOSE: Provide operational functions for import into main
#   - o1_train_model(model, dict_data_loaders, epoch, learnrate, type_loader, criterion)
#   - o2_model_backprop(model, data_loader, optimizer, criterion)
#   - o3_model_no_backprop(model, data_loader, criterion)
#   - o4_control_model_grad(model, control=False)
#   - o5_plot_training_history(model_name, model_hyperparameters)
#   - o6_predict_data(model, data_loader, dict_data_labels, dict_class_labels, topk=5)
#   - o7_show_prediction(data_dir, dict_prediction_results)
##

# Import libraries
import matplotlib.pyplot as plt
import time, os, random
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets, models
from torch import nn, optim
from PIL import Image


def o1_train_model(model, train_loader, valid_loader, epoch, decay, model_hyperparameters, criterion):
    '''
    Purpose:
        - Receive a model and start or continue training on it for e epochs
    Parameters:
        - model = inputted model (can be loaded with training history)
        - train_loader = data loader for training data for iterating
        - valid_loader = data loader for validation data for iterating
        - epoch = number of epochs to train
        - model_hyperparameters = dictionary of model hyperparameter information
        - criterion = the loss calculation method
    Returns:
        - model = model after e epochs of training
        - model_hyperparameters = revised hyperparameters for model after training
    '''
    # Print the GPU information or indicate that the GPU is not available if there is an issue
    print('Using GPU =', torch.cuda.get_device_name(), round(torch.cuda.get_device_properties(0).total_memory*(10**-9)), 'GB'\
                    if torch.cuda.is_available() else "WARNING GPU UNAVAILABLE")

    # Initialize a reference start time and subtract previous training time from reference point
    # Document starting learnrate and initialize a Boolean running variable to track deeper layer training
    t0 = time.time() - model_hyperparameters['training_time']*60
    startlearn = model_hyperparameters['learnrate']
    running = False

    # Set the optimizer for backpropogation. All parameters are set so that when unfrozen, they are included in backprop
    # NOTE: optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay= wd) to access unfrozen params
    optimizer = optim.Adam(model.parameters(), lr=model_hyperparameters['learnrate'], weight_decay=model_hyperparameters['weightdecay'])

    # Run the requested number of epochs worth of training by iterating e through the range of epochs
    for e in range(epoch):
        # Call backprop function with training data and model to return an updated model and the epoch training loss
        # Call validation function (no backprop) with the validation data and model and return performance data
        model, ave_training_loss = o2_model_backprop(model, train_loader, optimizer, criterion)
        val_count_correct, ave_validate_loss = o3_model_no_backprop(model, valid_loader, criterion)

        # Update the training history log with the average training loss and validation loss for this epoch
        model_hyperparameters['training_loss_history'].append(ave_training_loss)
        model_hyperparameters['validate_loss_history'].append(ave_validate_loss)

        # Print the epoch loss, accuracy, GPU usage, and runtime data. Accuracy is total correct over total in data
        print('Epoch: {}/{}..'.format(e+1, epoch),
            'Train Loss: {:.3f}..'.format(ave_training_loss),
            'Valid Loss: {:.3f}..'.format(ave_validate_loss),
            'Valid Accy: {:.2f}..'.format(val_count_correct / len(valid_loader.dataset)),
            'Mem: {:.2f}GB..'.format(np.around(torch.cuda.memory_allocated()*(10**-9), decimals=2)),
            'Time: {:.0f}min'.format((time.time() - t0)/60))

        # Reassigned model_hyperparameters['training_loss_history'] for this section to tlh for readability
        tlh = model_hyperparameters['training_loss_history']
        # This next section determines when to adjust learning based on training progress
        # This section is mainly a for fun exercise to tune a math algorithm for deciding when to adjust training
        # Hold loop until training_loss_history has enough elements to satisfy search requirements
        if len(tlh) > 3: # NOTE: 2
            # Compute reference: 3 times the first training loss factored by the current learnrate and the decay squared
            # Compute progress in training: the average of the last 2 training loss slopes
            # If progress in training is inverted sufficient enough to be greater than the reference, decay learnrate
            if 3*model_hyperparameters['learnrate']*decay*decay*tlh[0] < np.mean([tlh[-1]-tlh[-2], tlh[-2]-tlh[-3]]):
                model_hyperparameters['learnrate'] *= decay # multiply learnrate by the decay hyperparameter
                optimizer = optim.Adam(model.parameters(), lr=model_hyperparameters['learnrate'], weight_decay=model_hyperparameters['weightdecay']) # revise the optimizer to use the new learnrate
                print('Learnrate changed to: {:f}'.format(model_hyperparameters['learnrate']))
            # Compute reference: starting learnrate factored by decay^(9*decay^3))
            # Once learnrate has decayed to less than this value, call control_model_grad to activate deep layer training
            # Don't call if deep layer training has already been activated, set running to True to begin counting
            # In practice this performed well for various models and for overfitting vs regular training
            if model_hyperparameters['learnrate'] <= startlearn*decay**(9*(decay**3)) and model_hyperparameters['running_count'] == 0:
                model = o4_control_model_grad(model, True)
                model_hyperparameters['epoch_on'] = e
                running = True
            # If running, add to model running count to track the number of epochs run
            if running:
                model_hyperparameters['running_count'] +=1
            # Once the deep layers have trained for 20 epochs, call control_model_grad to deactivate deep layer training
            # Set the running variable to False to stop counting and prevent recalling deactivate layers
            if running and model_hyperparameters['running_count'] > 20:
                model = o4_control_model_grad(model, False)
                running = False
            # Find the basename of the loader's file root and check if it is overfit data
            # If overfit data, see if the train loss has gone below a target. If so end and print success
            # If the train loss has not gone below the target and the epochs have elapsed, print failure
            if os.path.basename(train_loader.dataset.root) == 'overfit':
                if np.mean([tlh[-1], tlh[-2], tlh[-3]]) < 0.0001:
                    print('\nModel successfully overfit images\n')
                    return model, model_hyperparameters
                if e+1 == epoch:
                    print('\nModel failed to overfit images\n')

    # Document the training time for the model and return the trained model and hyperparameters
    model_hyperparameters['training_time'] = np.around((time.time() - t0)/60, decimals=1)
    return model, model_hyperparameters


def o2_model_backprop(model, data_loader, optimizer, criterion):
    '''
    Purpose:
        - Conduct backpropogation on a model for data from a dataloader
    Parameters:
        - model = inputted model
        - data_loader = generator for data to provide model training
        - optimizer = defined optimizer for backpropogation
        - criterion = the loss calculation method
    Returns:
        - model = model after cycling through the data_loader (one epoch of training)
        - ave_training_loss = averaged training loss per batch of data
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Set device to GPU if available
    torch.cuda.empty_cache() # refresh GPU memory before starting
    model.to(device) # Move model to device
    epoch_train_loss = 0 # initialize total training loss for this epoch
    model.train() # Set model to training mode to activate regularizations such as dropout

    for images, labels in data_loader: # cycle through training data to conduct backpropogation
        images, labels = images.to(device), labels.to(device) # move data to GPU

        optimizer.zero_grad() # clear gradient history
        log_out = model(images) # run images through model to get logarithmic probability
        loss = criterion(log_out, labels) # calculate loss (error) for this image batch based on criterion

        loss.backward() # backpropogate gradients through model based on error
        optimizer.step() # update weights in model based on calculated gradient information
        epoch_train_loss += loss.item() # add training loss to total train loss this epoch, convert to value with .item()

    ave_training_loss = epoch_train_loss / len(data_loader.dataset) # determine average loss per training image
    return model, ave_training_loss # return the updated model and the average training loss


def o3_model_no_backprop(model, data_loader, criterion):
    '''
    Purpose:
        - Use the model to conduct predictions using the model
        - Return performance of the predictions across the data
    Parameters:
        - model = inputted model
        - data_loader = generator for data to conduct predictions
        - criterion = the loss calculation method
    Returns:
        - val_count_correct = number of correctly predicted data items
        - ave_validate_loss = averaged criterion loss per batch of data
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Set device to GPU if available
    torch.cuda.empty_cache() # refresh GPU memory before starting
    model.to(device) # Move model to device
    epoch_valid_loss = 0 # initialize total validate loss for this epoch
    val_count_correct = 0 # initialize total correct predictions on valid set
    model.eval() # set model to evaluate mode to deactivate generalizing operations such as dropout and leverage full model

    with torch.no_grad(): # turn off gradient tracking and calculation for computational efficiency
        for images, labels in data_loader: # cycle through validate data to observe performance
            images, labels = images.to(device), labels.to(device) # move data to GPU

            log_out = model(images) # run images through model to get logarithmic probability
            loss = criterion(log_out, labels) # calculate loss (error) for this image batch based on criterion
            epoch_valid_loss += loss.item() # add validate loss to total valid loss this epoch, convert to value with .item()

            out = torch.exp(log_out) # obtain probability from the logarithmic probability calculated by the model
            highest_prob, chosen_class = out.topk(1, dim=1) # obtain the top classes and probabilities from the output
            equals = chosen_class.view(labels.shape) == labels # determine how many correct matches were made in this batch
            val_count_correct += equals.sum()  # add the count of correct matches this batch to the total number this epoch

        ave_validate_loss = epoch_valid_loss / len(data_loader.dataset) # determine average loss per validate image
    return val_count_correct, ave_validate_loss # return this epoch's total correct predictions and average training loss


def o4_control_model_grad(model, control=False):
    '''
    Purpose:
        - Input a model and control active gradients on parameters at various layer depths
        - Print which layes have been controlled
    Parameters:
        - model = inputted model
        - control = whether to activate or deativate gradients
    Returns:
        - model = edited model with controlled layers
    '''
    # NOTE: Don't use model.children for network_depth, as this does not capture sublayers!
    network_depth = len(list(model.modules())) # Obtain the length of the layers used in the network
    param_freeze_depth = network_depth // 3 # Define what fraction of the network will be frozen and unfrozen
    controlled_layers = [] # Initialize the controlled layers list that will track what is frozen and unfrozen
    layer_depth = 0 # Initialize the start for iterating through layers

    for layer in list(model.modules()): # Iterate through layers in the model
        layer_depth += 1 # Increase current layer depth by 1 to progress through network layers

        if (network_depth - param_freeze_depth) <= layer_depth: # Once sufficiently deep, control layers
            controlled_layers.append(layer._get_name()) # Add current layer's name to list of controlled layers
            for param in layer.parameters(): # Iterate through the parameters in this layer
                param.requires_grad = control # Freeze or unfreeze the gradient on the parameter

        if layer._get_name() == 'Linear': # The fully connected layers are always unfrozen
            for param in layer.parameters(): # Iterate parameters
                param.requires_grad = True # Set gradient to true

    print(f'\n Toggle requires_grad = {control}: ', controlled_layers, '\n') # Print changes made to active grads
    return model # Return model with changed param activity


def o5_plot_training_history(model_name, model_hyperparameters, file_name_scheme, train_type='loaded'):
    '''
    Purpose:
        - Plot the training and validation loss history for the inputted model
        - Plot lines indicating when layers were activated and deactivated if controlled
        - Save the plot with the name according to the type of training, skip if a loaded version
    Parameters:
        - model_name = name of model, used for title on plot
        - model_hyperparameters = contains the history for plotting
        - file_name_scheme = directory and naming convention for loading
        - train_type = offer control on saving
    Returns:
        - none
    '''
    # Plot training history information
    plt.clf()
    plt.plot(model_hyperparameters['training_loss_history'], label='Training Training Loss')
    plt.plot(model_hyperparameters['validate_loss_history'], label='Validate Training Loss')

    # If deep layer training has started, plot dotted lines for start and finish
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

    # Plot title and labels
    plt.title(model_name)
    plt.ylabel('Total Loss')
    plt.xlabel('Total Epoch ({})'.format(len(model_hyperparameters['training_loss_history'])))
    plt.legend(frameon=False)

    # If the plot was not loaded, save the plot using the naming convention
    if train_type != 'loaded':
        plt.savefig(file_name_scheme + '_training_history_' + train_type + '.png')
        print('Saved', train_type, 'training history to project directory')

    # Show plot and unblock to allow function continuation, pause to load image and avoid from freezing
    plt.show(block=False)
    plt.pause(2)
    plt.clf()
    plt.close()

def o6_predict_data(model, data_loader, dict_data_labels, dict_class_labels, topk=5):
    '''
    Purpose:
        - Compute probabilities for various classes for an image using a model
    Parameters:
        - model = trained deep neural net for computation
        - data_loader = generator for data items to be iterated through for parallel prediction
        - dict_data_labels = dictionary containing the names of each class for the data indexes
        - dict_class_labels = dictionary containing the class indexes for the data indexes
        - topk = number of class outputs
    Returns:
        - dict_prediction_results = dictionary containing predictions and probabilities for data keys
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # set device to GPU if available
    torch.cuda.empty_cache() # refresh GPU memory before starting
    model.to(device) # move model to device
    dict_prediction_results = {} # initialize the prediction results dictionary
    model.eval() # set model to evaluate mode to deactivate generalizing operations such as dropout and leverage full model
    if len(dict_class_labels) < topk: # confirm there are enough different classes to satisfy results ranking
        topk = len(dict_class_labels) # replace number of ranked results to the number of classes if not

    with torch.no_grad(): # turn off gradient tracking and calculation for computational efficiency
        for image, filenames in data_loader: # cycle through data for inference
            image = image.to(device) # move data to GPU

            log_out = model(image) # run images through model to get logarithmic probability
            model_output = torch.exp(log_out) # obtain probability from the logarithmic probability
            probabilities, class_indexes = model_output.topk(topk, dim=1) # obtain the top results

            for index in np.arange(len(filenames)):# iterate through filenames batch with index
                # Find the class prediction name by comparing the class label dictionary to the data label dictionary
                if dict_data_labels:
                    class_prediction = [dict_data_labels[dict_class_labels[value]] for value in class_indexes.tolist()[index]]
                else:
                    class_prediction = [dict_class_labels[value] for value in class_indexes.tolist()[index]]
                # Then add this filename to the prediction results dictionary with the corresponding results and
                dict_prediction_results[filenames[index]] = [class_prediction, probabilities.tolist()[index]]
    return dict_prediction_results # Return


def o7_show_prediction(data_dir, dict_prediction_results):
    '''
    Purpose:
        - Randomly choose a piece of data from the predict folder
        - Display the chosen data and display the class outputs and corresponding probabilities
    Parameters:
        - data_dir = pathway to the data directory containing the data of interest
        - dict_prediction_results = dictionary of the prediction results on the dataset of interest
    Returns:
        - none
    '''
    # Randomly select an example file to conduct a prediction
    example_prediction = random.choice(list(dict_prediction_results.keys()))

    # Open and show the prediction
    plt.imshow(Image.open(data_dir + 'predict/' + example_prediction)); # no need to process and inverse transform, our data is coming from the same path, I'll just open the original
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    # Plot the predicted class, the probabilities, and use the data's filename for the title
    plt.bar(dict_prediction_results[example_prediction][0], dict_prediction_results[example_prediction][1])
    plt.title(example_prediction)
    plt.xticks(rotation=20);
    plt.show(block=False)
    plt.pause(3)
    plt.close()
