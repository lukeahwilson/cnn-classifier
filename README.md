# convolutional-classifier
A convolutional network leveraging the PyTorch framework and associated libraries and pretrained models to train a classifier for demonstrating the many practical applications of this technology.

## PURPOSE:
API to train and apply leveraged pretrained vision models for classification

## REQUIREMENTS:
- Pretained model is downloaded and can be trained on a dataset by user
- The number of attached fully connected layers is customizable by the user
- The deeper convolutional layers are unfrozen for a period of time during training for tuning
- User can load a model and continue training or move directly to inference
- Saved trained model information is stored in a specific folder with a useful naming convention
- There are time-limited prompts that allow the user to direct processes as needed
- Training performance can be tested before moving onward to inference if desired
- Predictions are made using paralleled batches and are saved in a results dictionary

## HOW TO USE:
- If no model has been trained and saved, start by training a model
- Store data in folders at this location: os.path.expanduser('~') + '/Programming Data/'
- For training, 'train' and 'valid' folders with data are required in the data_dir
- For overfit testing, an 'overfit' folder with data is required in the data_dir
- For performance testing, a 'test' folder with data is required in the data_dir
- For inference, put data of interest in a 'predict' folder in the data_dir
- For saving and loading models, create a 'models' folder in the data_dir

## Table Of Contents

### cnn_main
- Retrieve command line arguments to dictate model type, training parameters, and data
- Load image datasets, process the image data, and convert these into data generators
- Create a default naming structure to save and load information at a specified directory
- Download a pretrained model using input arguments and attach new fully connected output Layers
- Define criterion for loss, if training is required by the input arg, execute the following:
    o Prompt user for overfit training, if yes, initiate training against pretrained features
    o Prompt user for complete training, if yes, initiate training against pretrained features
    o Save the hyperparameters, training history, and training state for the overfit and full models
- If training is no requested by the input arg, execute the following:
    o Load in a pretrained model's state dict and it's model_hyperparameters
    o Display the training history for this model
- Provide prompt to test the model and perform and display performance if requested
- Provide prompt to apply the model towards inference and put model to work if requested
- Show an example prediction from the inference

### cnn_model_functions
- Classifier(nn.Module)
- m1_create_classifier(model_name, hidden_layers, classes_length)
- m2_save_model_checkpoint(model, file_name_scheme, model_hyperparameters)
- m3_load_model_checkpoint(model, file_name_scheme)

### cnn_operational_functions
- o1_train_model(model, dict_data_loaders, epoch, learnrate, type_loader, criterion)
- o2_model_backprop(model, data_loader, optimizer, criterion)
- o3_model_no_backprop(model, data_loader, criterion)
- o4_control_model_grad(model, control=False)
- o5_plot_training_history(model_name, model_hyperparameters)
- o6_predict_data(model, data_loader, dict_data_labels, dict_class_labels, topk=5)
- o7_show_prediction(data_dir, dict_prediction_results)

### cnn_utility_functions
- u1_get_input_args()
- u2_load_processed_data(data_dir)
- u3_process_data(transform_request)
- u4_data_iterator(dict_datasets)
- u5_time_limited_input(prompt, default=True)
- u6_user_input_prompt(prompt, default)

## Credits
This repository was built on the knowledge acquired from the high-quality and indepth lectures, assignments, and projects that I completed in attendance of Udacity's paid nanodegree program 'AI Programming with Python'. This repository also used design ideas and deep learning fundamentals presented from Stanford's online lecture series CS231n Convolutional Neural Networks for Visual Recognition.

## Dependencies
Please see the `requirements.txt` file or the environment `env-classifier-pytorch.yaml` file for minimal dependencies required to run the repository code.

## Install
To install these dependencies with pip, you can issue `pip3 install -r requirements.txt`
To install these dependencies with conda, use `conda env create --file env-classifier-pytorch.yaml`
