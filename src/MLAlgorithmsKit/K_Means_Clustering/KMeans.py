import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

class K_means:
    
    def train(self,n_iter = 2000):
        """'train' method takes argument n_iter(no.of Iterations) and returns nothing. It has been initialised with a default value of                   2000."""
        self.n_iter = n_iter
    
    def test_train_split(self,X,y,size):
        """Splits the dataset into trainig set and test set based on input split fraction.
        takes X,y,test_size as arguments and returns X_train,X_test,y_train,y_test."""
        m_test = int(X.shape[0]*size)
        X_test = X[0:m_test,:]
        y_test = y[0:m_test]
        X_train = X[m_test:,:]
        y_train = y[m_test:]
        return X_train,X_test,y_train,y_test  
    
    
    def predict(self,X,k):
        
        """ It takes X,k(no.of clusters) as argument and returns y_pred as predicted by the model."""
        
        m = X.shape[0]
        n = X.shape[1]
        
        # Normalize.
        for i in range(X.shape[1]):
            X[:,i] = (X[:,i] - np.mean(X[:,i]))/(np.std(X[:,i]) + np.exp(-9))
        ran = random.sample(range(m),k)
        centr = np.zeros((k,n))
        y_pred = np.zeros((m,1))
        
        # Random Centroid Initialisation.
        for i in range(k):
            centr[i:i+1,:] = X[ran[i]:ran[i]+1,:]

        # Algorithm.
        for j in range(self.n_iter):
            freq = np.zeros((k,1))
            for i in range(m):
                dis = X[i:i+1,:] - centr
                dis = dis**2
                
                # Distance Calculation.
                dis = np.sum(dis,axis =1)
                
                # Finding cllosest centroid.
                y_pred[i,0] = np.argmin(dis)
            centr = np.zeros((k,n))  
            
            y_pred.astype('int64')
            
            # Centroid Update.
            for i in range(m):
                r = int(y_pred[i,0])
                centr[r:r+1,:] += X[i:i+1,:]
                freq[r,0]+=1
            for i in range(k):
                centr[i:i+1,:]/=freq[i,0]
        return y_pred.flatten()
