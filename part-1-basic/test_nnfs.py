import numpy as np
import nnfs
from nnfs.datasets import spiral_data

#nnfs.init()



import matplotlib.pyplot as plt

X,y= spiral_data(samples=100,classes=3)
print(X)
plt.scatter(X[:,0],X[:,1],c= y, cmap= 'brg')
plt.show()