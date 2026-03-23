import numpy as np

input_vector = np.array([2, 4, 11])

print("one dimensional vector : ", input_vector)


#Transform input_vector to 2D column vector
input_vector = np.array(input_vector, ndmin=2).T
print("\n The input vector : \n", input_vector)
print("\n Shape of vector :  \n", input_vector.shape)

#Unity function
number_of_samples = 1200
low = -1
high = 0
s = np.random.uniform(low, high, number_of_samples)
print("Samples : \n", s)

#all values of s are within the half open interval [-1, 0):
print(np.all(s > -1) and np.all (s < 0))

from scipy.stats import truncnorm

def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return (truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd))
#create wih matrix
no_of_input_nodes = 3
no_of_hidden_nodes = 4
rad = 1 / np.sqrt(no_of_input_nodes)

X = truncated_normal(mean=2, sd=1, low=-rad, upp=rad)

wih = X.rvs((no_of_hidden_nodes, no_of_input_nodes))

print("\nwih - matrix : \n", wih)

#create who matrix
no_of_hidden_nodes = 4
no_of_output_nodes = 2
rad = 1 / np.sqrt(no_of_hidden_nodes)

who = X.rvs((no_of_output_nodes, no_of_hidden_nodes))

print("\n who matrix : \n", who)
