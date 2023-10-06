import numpy as np



a = [1,2,3]

## row vector 

# convert to array 
print(np.array([a]))

#convert to array 
print(np.expand_dims(np.array(a), axis=0))


## column vector 

b = [2,3,4]

# transposed
print(np.array([b]).T)

a_array = np.array([a])
b_array = np.array([b]).T

print(np.dot(a_array,b_array))
