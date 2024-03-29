import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Linear_Regression:
   
        
  def train(self,alpha=0.05,lam=0.01,n_iter=6000):
    """  'train' method takes argument as Learning Rate(alpha), Regularisation Constant(lam), and No.of Iterations(n_iter).
     All of these values have been initialised by a default value, but can be changed when required.
     J_hist and noi(number of iterations) keeps track of cost along with iteration. """

    self.alpha = alpha
    self.n_iter = n_iter
    self.lam = lam 
    self.J_hist = []
    self.noi = []
   
 
  
  def cost(self):
    """ cost method takes no arguments and returns the mean squared error of our hypothesis. """
    h = self.X@self.theta
    return (1/(2*self.m))*np.sum((h-self.y)**2) + (self.lam/(self.m*2))*np.sum(self.theta[1:,0]**2)
  
  
  
  def fit(self,X,y):
    """  'fit' takes X_train,y_train as arguments, applies gradient descent algorithm to fit our parameters theta and returns theta. """
    self.m = X.shape[0]
    self.n = X.shape[1]
    self.theta = np.zeros((self.n + 1,1))
    
    #  Normalization
    self.X = self.normalize(X)
    self.y = y[:,np.newaxis]

    # Gradient Descent Algorithm.
    for i in range(self.n_iter):
         # Parameter update
         theta1  = self.theta
         theta1[0,0] = 0
         self.theta = self.theta - (self.alpha/self.m) * ((self.X.T @ (self.X @ self.theta - self.y)) + self.lam*theta1)
         self.J_hist.append(self.cost())
         self.noi.append(i)
         if(i==0):
           print("Initial Cost:",self.cost())
         if(i==self.n_iter-1):
           print("Final Cost:",self.cost())  
  
  
  def score(self,X,y):
    """'score' takes X,y as argument and returns the score of our prediction."""
    X = self.normalize(X)
    y = y[:,np.newaxis]
    y_pre = X@self.theta
    return  1 - (((y - y_pre)**2).sum() / ((y - y.mean())**2).sum())
  
 
  def accuracy(self,y_test,y_pred):
    m = len(y_test)
    sum1 =0
    for i in range(m):
     if(y_test[i]==y_pred[i]):
      sum1+=1
    return (sum1/m)*100  
    
  
  def plot_learn(self):
    """plots the learning curve; cost function vs no.of iterations."""
    plt.plot(self.noi,self.J_hist)
    plt.xlabel("Number Of Iterations")
    plt.ylabel("Cost Function")
    plt.title("Const Function vs Iteration")

  
  def test_train_split(self,X,y,size):
        """Splits the dataset into training set and test set based on the input split fraction.
         It takes X,y,test_size as arguments and returns X_train,X_test,y_train,y_test"""
        m_test = int(X.shape[0]*size)
        X_test = X[0:m_test,:]
        y_test = y[0:m_test]
        X_train = X[m_test:,:]
        y_train = y[m_test:]
        return X_train,X_test,y_train,y_test   
 

  def predict(self,X):
    """ predicting value of target feature using the trained model.
          takes X as argument and returns as prediction done by the model.
      """
    m = X.shape[0]
    X = self.normalize(X)
    y_pred = X@self.theta
    return y_pred.flatten()

  # Normalises X
  def normalize(self,X):
    m = X.shape[0]
    for i in range(X.shape[1]):
        X[:,i] = (X[:,i] - np.mean(X[:,i]))/(np.std(X[:,i]) + np.exp(-9))
    X = np.hstack((np.ones((m, 1)), X))
    return X

  #  returns the parameter theta(weight) of the model.
  def get_params(self):
      return self.theta

  


