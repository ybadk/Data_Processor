import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate the data
X = np.arange(1, 101).reshape(-1, 1)  # Generate a sequence from 1 to 100
y = 4 + 3 * X + np.random.randn(100, 1)  # Target values with some random noise

# Create the design matrix
X_b = np.c_[np.ones((100, 1)), X]

# Define the cost function to minimize
def cost_function(theta):
    theta0, theta1 = theta
    predictions = theta0 + theta1 * X_b[:, 1]
    return np.sum((predictions - y) ** 2)

# Initialize the parameters
theta_init = [4, 3]

# Minimize the cost function using the minimize function from scipy
res = minimize(cost_function, theta_init)

# Get the optimized parameters
theta_best = res.x

print("Theta best:", theta_best)

X_new = np.array([[1, 0], [1, 2]])
y_predict = X_new.dot(np.append([1], theta_best))
print("Predicted values for new data:", y_predict)

# Use sklearn to perform linear regression
lin_reg = LinearRegression()
lin_reg.fit(X_b, y)
print("Intercept:", lin_reg.intercept_)
print("Coefficients:", lin_reg.coef_)

# Optionally plot the results for visualization
X = np.arange(1, 101).reshape(-1, 1)  # For plotting purposes
plt.plot(X, y, "b.", label="Actual Data")  # Blue dots for actual data points
plt.scatter(X_new[:, 0], y_predict, color='r', label="Predicted Points")  # Red dots for predicted new
data points
plt.legend()
plt.show()
