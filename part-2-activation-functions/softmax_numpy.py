import numpy as np
# Values from the earlier previous when we described
  # what a neural network is
layer_outputs = [4.8, 1.21, 2.385]
  # For each value in a vector, calculate the exponential value
exp_values = np.exp(layer_outputs) 
print('exponentiated values:') 
print(exp_values)
  # Now normalize values
norm_values = exp_values / np.sum(exp_values) 
print('normalized exponentiated values:') 
print(norm_values)
print('sum of normalized values:', np.sum(norm_values))


# Get unnormalized probabilities
#exp_values = np.exp(inputs)
  # Normalize them for each sample
#probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)