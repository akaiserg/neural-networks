import numpy as np 
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

from Util.LayerDense import Layer_Dense
from Util.ActivationRelu import Activation_Relu
from Util.ActivationSoftMax import Activation_Softmax
from Util.Cross_Entropy import LossCategoricalCrossEntropy

X, y = spiral_data(samples=100, classes=3) # 100 is multiplied by 3 

#print(np.shape(X))

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_Relu()


# Create second Dense layer with 3 input features (as we take output # of previous layer here) and 3 output values (output values) 
dense2 = Layer_Dense(3, 3)

# Create Softmax activation (to be used with Dense layer):
activation2 = Activation_Softmax()

# Create loss function
loss_function = LossCategoricalCrossEntropy()


# Perform a forward pass of our training data through this layer
dense1.forward(X)


# with this activation function the negative numbers are converted into 0
activation1.forward(dense1.output)

# Make a forward pass through second Dense layer
# it takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)


# Make a forward pass through activation function
# it takes the output of second dense layer here
activation2.forward(dense2.output)


# Let's see output of the first few samples:
print(activation2.output[:5])


# Perform a forward pass through loss function
# it takes the output of second dense layer here and returns loss 
loss = loss_function.calculate(activation2.output, y)

# Print loss value
print('loss:', loss)


# Calculate accuracy from output of activation2 and targets # calculate values along first axis
predictions = np.argmax(activation2.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1) 
accuracy = np.mean(predictions==y)
  # Print accuracy
print('acc:', accuracy)

# # the result of 300x2 dotproduct 2x3  = dense1.output
# # Let's see output of the first few samples:
# print(dense1.output[:5])

# #print(X)
# #plt.scatter(X[:, 0], X[:, 1], c= y, cmap= 'brg')
# #plt.show()


# print("###############################")

# # with this activation function the negative numbers are converted into 0
# activation1.forward(dense1.output)

# print(activation1.output[:5])


# # Plot the activation function
# plt.plot(dense1.output, activation1.output, label='ReLU Activation Function')
# plt.title('ReLU Activation Function')
# plt.xlabel('Input')
# plt.ylabel('Output')
# plt.grid(True)
# plt.legend()
# plt.show()