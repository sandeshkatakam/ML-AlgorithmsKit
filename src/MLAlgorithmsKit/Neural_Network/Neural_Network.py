import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Neural_Network:
    def train(self,alpha = 0.07,n_iter = 2000,lam = 0.20):
       
        """'train' method takes argument as Learning Rate(alpha), Regularisation Constant(lam), and No.of Iterations(n_iter).
            All of these values have been initialised by a default value, but can be changed when required.
            J_hist and noi(number of iterations) keeps track of cost along with iteration. """
                 
        self.alpha = alpha
        self.n_iter = n_iter
        self.lam = lam
        self.J_hist = []
        self.noi = []
        self.theta = []
    
    #definition of sigmoid.    
    def sigmoid(self,x):
        return  1/(1+np.exp(-x))
    
    def fit(self,X,y,k,layer):
        """ It takes X,y,k(No.of classes),layers(Hidden layers as tuple) and sets the parameter theta for the model."""
        
        m = X.shape[0]
        n = X.shape[1]
        # Normalization.
        X = self.normalize(X)
        Y = np.zeros((m,k))
        
        #One-vs-All matrix.
        for i in range(m):
            Y[i,y[i]] = 1
        le = len(layer)
        self.le = le
        #Random Initialisation of theta.
        for i in range(le+1): 
            if i==0:
                self.theta.append(np.random.randn(layer[i],n+1))
            elif i<le:
                self.theta.append(np.random.randn(layer[i],layer[i-1]+1))
            elif i==le:
                self.theta.append(np.random.randn(k,layer[i-1]+1))
        no=0
        cost=0
        
        #Algorithm.
        for j in range(self.n_iter):
            if(j==1):
                print("Initial Cost: ",cost)
            if(j==self.n_iter-1):
                print("Final Cost: ",cost)
            # taking mini - batch as 100.
            for t in range(m//100): 
                grad = []
                active = []
                delta = []
                cost=0
                # Initialisaton of Gradient,Delta for each layer.
                for i in range(self.le+1):
                  if i==0:
                      grad.append(np.zeros((layer[i],n+1)))
                      delta.append(np.zeros((layer[i],n+1)))
                  elif i<le:
                      grad.append(np.zeros((layer[i],layer[i-1]+1)))
                      delta.append(np.zeros((layer[i],layer[i-1]+1)))
                  elif i==le:
                      grad.append(np.zeros((k,layer[i-1]+1)))
                      delta.append(np.zeros((k,layer[i-1]+1)))                    
                
                # Forward Propogation.
                for i in range(self.le+1):
                    yk = Y[t*100:t*100 + 100,:]
                    yk = yk.T
                    if(i==0):
                      a1 = X[t*100:t*100 + 100,:]
                      a1 = a1.T
                    else:
                      a1 = a2
                    thet = np.array(self.theta[i])
                    z2 = thet@a1
                    a2 = self.sigmoid(z2)
                    
                    if(i!=le):
                        a2 = np.vstack((np.ones((1,100)),a2))
                    # Appending activation in active.
                    active.append(a2)
                h = active[-1]
                
                # Cost Calculation.
                cost+= -1*(np.sum(yk*np.log(h) + (1-yk)*np.log(1-h))/100)
                for i in range(self.le+1):
                    thet = self.theta[i]
                    cost += ((self.lam/200)*np.sum(thet[:,1:]**2))
                no+=1
                self.J_hist.append(cost)
                self.noi.append(no)
                
                # Back Propogation.
                # delta for output layer.
                delta_l = h-yk
                
                for i in range(self.le,0,-1):
                    thet = np.array(self.theta[i])
                    
                    a1 = active[i-1]
                    
                    if i==le:
                        delt2 = delta_l
                    else:
                        delt2 = delt1
                    
                    # ignoring bias term.
                    a1_ = a1[1:,:]
              
                    delt1 = ((thet[:,1:].T@delt2))*((a1_)*(1-a1_))
                    # delta update.
                    delta[i] += np.dot(delt2,a1.T)
                                          
                       
                delta[0] += (delt1@(X[t*100:t*100 + 100,:]))
                
                #Gradient Calculation
                for i in range(self.le+1):
                    thet = self.theta[i]
                    thet[:,0] = 0
                    grad[i] = (delta[i] + self.lam*thet)/100
                    # theta update.
                    self.theta[i] = self.theta[i] - (self.alpha*grad[i])                       
            
    def predict(self,X):
        """ this takes X as argument and returns y_pred as predicted by the model."""
        m = X.shape[0]
        n = X.shape[1]
        X = self.normalize(X)
        y_pred = np.zeros((m,1))
        yp = 0
        for j in range(m//100):
            for i in range(self.le+1):
                thet = self.theta[i]
                if i==0:
                    a1 = X[j*100:j*100 + 100,:]
                    a1 = a1.T
                else:
                  a1=a2
                z2 = thet@a1
                a2 = self.sigmoid(z2)
                if i!=self.le:
                    a2 = np.vstack((np.ones((1,100)),a2))
                if i==self.le:
                    yp = a2
            y_pred[j*100:j*100 + 100] = np.argmax(yp,axis=0).reshape((100,1))
        return y_pred.flatten()
    
    # Calculates Accuracy.
    def accuracy(self,y_pred,y_test):
       m = len(y_pred)
       sum1=0
       for i in range(m):
           if(y_pred[i]==y_test[i]):
               sum1+=1
       return (sum1/m)*100    
   
    def plot_learn(self):
      """ Plots the learning curve; cost function vs no.of iterations."""
      plt.plot(self.noi,self.J_hist)
      plt.xlabel("Number Of Iterations")
      plt.ylabel("Cost Function")
      plt.title("Cost Function vs Iteration")
       
    #Normalization.
    def normalize(self,X):
        m = X.shape[0]
        for i in range(X.shape[1]):
            X[:,i] = (X[:,i] - np.mean(X[:,i]))/(np.std(X[:,i]) + np.exp(-9))
        X = np.hstack((np.ones((m, 1)), X))
        return X