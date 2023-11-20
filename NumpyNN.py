import numpy as np
from pprint import pprint

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def initialise_network(n_inputs, n_hidden, n_outputs):
 
    network = list()
    hidden_layer = np.random.randn(n_hidden, n_inputs + 1)
    network.append(hidden_layer)
    output_layer = np.random.randn(n_outputs, n_hidden + 1)
    network.append(output_layer)
    return network
        
# Forward Propagation
def activate(weights, inputs):
    activation = np.dot(weights, inputs)
    return activation

def activation_function(activation):
    return sigmoid(activation)

def forward_propagate(network, row):
    inputs = row

    for layer in network:
        new_inputs = []
        for new_layer in layer:
            activation = activate(new_layer, inputs)
            activation = activation_function(activation)
            new_inputs.append(activation)
        inputs = new_inputs
    return inputs

# test forward propagation
network = initialise_network(2, 1, 2)
row = [1, 0, 1]
output = forward_propagate(network, row)
print(output)