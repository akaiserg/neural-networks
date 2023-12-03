import numpy as np
import matplotlib.pyplot as plt


inputs = [1.0, 2.0, 3.0, 2.5] 
weights = [0.2, 0.8, -0.5, 1.0] 
bias = 2.0

outputs = np.dot(weights, inputs) + bias

# Create a scatter plot of inputs (optional)
plt.scatter(range(len(inputs)), inputs, label="Inputs")

# Display the computed output as text
plt.text(len(inputs) - 1, outputs, f'Output = {outputs}', fontsize=12, ha='center')
plt.legend()
plt.xlabel('Input Index')
plt.ylabel('Value')
plt.title('Dot Product Operation')
plt.show()

print(outputs)