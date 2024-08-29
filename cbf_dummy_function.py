import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import erf

# This is a dummy function that represents the reward function of the CBF - use to plot the surface

def nonlinear_m(y):
    # Define a nonlinear relationship between y (obstacle size) and m (gaussian dist along cbf value mean).
    a, b, c = 0.1, 0.5, 2
    return 0.05*(y-5)**2        #a * y**2 + b * y + c

# Skew-normal distribution
def skew_normal_pdf(x, y):
    sigma = 0.2
    m = nonlinear_m(y)
    alpha = -0.1 # 0 = no skew <1 = left skew, >1 = right skew
    # Standard normal PDF
    phi = np.exp(-0.5 * ((x - m) ** 2) / (sigma ** 2))
    # Standard normal CDF for the skew part
    Phi = 0.5 * (1 + erf(alpha * (x - m) / (sigma * np.sqrt(2))))
    # Combine to get the skew-normal PDF
    return 3 * phi * Phi - 0.5

def evaluate_surface(x_values, y_values):
    # Evaluate the Gaussian function for given arrays of x and y values
    X, Y = np.meshgrid(x_values, y_values)
    #Z = gaussian_2d(X, Y)
    Z = skew_normal_pdf(X,Y)
    return X, Y, Z

def plot_surface(X, Y, Z):
    # Plot the surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('CBF Parameter Values')
    ax.set_ylabel('Obstacle Size')
    ax.set_zlabel('Expected reward')
    return fig, ax

# Example usage
x_values = np.linspace(0, 3, 100)
y_values = np.linspace(0, 3, 100)
X, Y, Z = evaluate_surface(x_values, y_values)
fig, ax = plot_surface(X, Y, Z)


# pltpts = np.array(([0.2,0.5],[0.5,1.5],[0.9,2.9]))
# a = pltpts[:,0]
# b = pltpts[:,1]

# c = skew_normal_pdf(a, b)
# ax.scatter(a,b,c,color='g', marker='o',s=200)
plt.show()




