import numpy as np
import nnfs
from nnfs.datasets import spiral_data

#nnfs.init()


# ReLU activation
class Activation_ReLU:
    # Forward pass
    def forward(self, inputs):
        # Calculate output values from input
        self.output = np.maximum(0,inputs)
        



# Dense layer
class Layer_Dense:

    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))


    # Forward pass
    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases


# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()

#print(f"weights {dense1.weights}")
#print(f"biases {dense1.biases}")

#print(f"inputs {X}")

# Perform a forward pass of our training data through this layer
dense1.forward(X)

activation1.forward(dense1.output)



# Let's see output of the first few samples:
#print("result")
#print(dense1.output[:5])

print(activation1.output[:5])
