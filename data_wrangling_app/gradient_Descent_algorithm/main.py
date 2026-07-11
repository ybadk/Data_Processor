import numpy as np

# Set hyperparameters
eta = 0.1  # Learning rate
n_iterations = 1000  # Number of iterations
m = 100  # Number of training examples

# Initialize theta randomly
theta = np.random.randn(2, 1)  # Adjust dimensions based on your actual data

for iteration in range(n_iterations):
    # Compute gradients
    gradients = (2 / m) * X.T.dot(X.dot(theta) - y)

    # Update theta using the computed gradients and learning rate
    theta -= eta * gradients

print(theta)
