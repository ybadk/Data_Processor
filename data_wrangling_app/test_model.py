from sklearn.datasets import make_blobs
from neural_networks1 import *
n_samples = 500
blob_centers = ([2, 6], [6, 2], [7, 7])
n_classes = len(blob_centers)

data, labels = make_blobs(n_samples=n_samples,
                          centers=blob_centers,
                          random_state=7)

import matplotlib.pyplot as plt

colours = ('green', 'red', "yellow")

fig, ax = plt.subplots()

for n_class in range(n_classes):
    ax.scatter(data[labels==n_class][:, 0],
               data[labels==n_class][:, 1],
               c=colours[n_class],
               s=40,
               label=str(n_class))
#plt.show()

print("Labels : ", labels[:10])

#one - hot representation of Labels
import numpy as  np
labels = np.arange(n_classes) == labels.reshape(labels.size, 1)
labels = labels.astype(np.float64)

print("\n one - hot transformed labels : \n", labels[:10])


#create a train and test dataset
from sklearn.model_selection import train_test_split

res = train_test_split(data, labels, train_size=0.8, test_size=0.2, random_state=42)
train_data, test_data, train_labels, test_labels = res
print("\nTrain Labels\n", train_labels[:10])


#create a neural network with two input nodes and three output nodes. one output node for each class
from neural_networks1 import NeuralNetwork

simple_network = NeuralNetwork(no_of_in_nodes=2,
                               no_of_out_nodes=3,
                               no_of_hidden_nodes=5,
                               learning_rate=0.3)


#train network with data and labels data from from training samples

for i in range(len(train_data)):
    simple_network.train(train_data[i], train_labels[i])

#simple_network.evaluate(train_data, train_labels)
