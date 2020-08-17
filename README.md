# PyTorch Image Classifier

A command line application and Jupyter notebook implementation of image classification leveraging PyTorch. This code allows a user to use a pretrained model, and modify it for use in predicting the correct label of an image. 

The dataset used is a variety of images containing flowers distributed across 102 classes. 

# utils.py 
All of the functions used in this CLI application are contained in this file.

# train.py
The user can utilize a variety of commands to customize the model creation and training process. Once training is complete the updated model is stored in a checkpoint file that can be used in prediction.

# predict.py 
Allows user to load a model from checkpoint file and predict label of an image using the trained model.
