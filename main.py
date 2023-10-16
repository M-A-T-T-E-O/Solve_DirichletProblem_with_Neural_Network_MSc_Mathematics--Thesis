# Modules
import torch as nn

# Function imported
from layer_class import PyTorchNN
from preprocessing import preprocessing
from trainingNN import trainingNN
from validationNN import validationNN

# Create, train and validate NN

# Initialize the pseudorandom generator with a seed
nn.manual_seed(1234)

# Implementation of the feed-forward Neural network
MyNN = PyTorchNN()

# Show the NN parameters before its training
print("\nThe NN parameters (weight and bias) before training: ", list(MyNN.parameters()))

# Run Dataset preprocessing
[x_train, y_train, x_test, y_test] = preprocessing()

# Run the NN training algorithm
trainingNN(MyNN, x_train, y_train, -1, 300000)

# Show the NN parameters after training phase
print(" The NN parameters (weight and bias) after training: ", list(MyNN.parameters()))

# Run validation NN
validationNN(MyNN, x_test, y_test)










