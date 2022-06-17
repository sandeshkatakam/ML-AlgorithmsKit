# This is code implementation of Cost function.
import numpy as np
def compute_cost(X,y,theta):
  """
  Calculates the cost function for given data

  Arguments:
  X : Input features passed in as vector or matrix(if multiple features)(use numpy)
  y : Labels or ground truth values passed in as a vector
  theta : parameters matrix for calculating the hypothesis function h.(should be passed as an array)
  
  Returns:
  The cost function for all the training examples in dataset

  """
  m = len(y)
  X = np.array([np.ones((m)), X])
  print(X.shape)
  h = theta.T * X # matrix multiplication
  C=[]
  C=h-y
  C=np.square(C)
  J=np.sum(C)/(2*m)
  return J


# Implementation of Gradient Descent Algorithm
def gradient_descent(X,y,theta,alpha, num_iters):
  """
  Performs the gradient descent for the Data given no of iterations

  Arguments:
  X: Input array of data (for uni-variate linear reg : dim = (m,2))
  y: labels
  theta: parameters matrix (for 2 parameters : dim = (2,1))
  Note: Dimensions of H : (m,1), Dimensions of y : (m,1) (Both should be equal)
  Returns:
  Theta values after the minimization process of loss function
  """
  J_history = np.zeros(num_iters, 1)
  m = len(y)
  for iters in range(num_iters-1):
    H = np.matmul(X,theta)
    J = np.square(H-y)
    K = (X.T) * J
    K = K/m
    # Theta update step(parameter update step)

    theta = theta - ((alpha)*K)
    J_history[iters] = compute_cost(X, y, theta)
    return [theta, J_history]


def mean_normalization(X):
    """
    Performs the feature normalization step for the Input data

    Arguments:
    X : Array of Input data
    How is X stacked? Dimensions of X
    Returns:
    Mean normalized input data( Mean value of the data will be zero)
    """
    mean_norm = (X - np.mean(X))/(np.max(X)- np.min(X))
    return mean_norm
