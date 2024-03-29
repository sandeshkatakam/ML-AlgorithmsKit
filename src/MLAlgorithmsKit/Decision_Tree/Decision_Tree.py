import numpy as np
import pandas as pd
import random

class Decision_Tree:
       
        """ This class builds,fits,predicts using data. The target variable should be present in the last column of data and all the
           features should be numerical."""
        
        def train(self):
        
            """ 'train' method takes no arguments. 'tree' is a dictionary to store the splits with depth of tree.'splits' is a dictionary which holds the combination of all possible splits(each feature and its each unique value) """ 
         
            self.tree={}
            self.splits = {}
    
    
   
        def fit(self,df, flag=0):
            
            """ 'fit' is a recursive function which builds the tree. It takes the training data as argument and returns the tree built. """
        
        
            if flag == 0:
                
                # column_head stores all the name of features from the given dataset.
                
                global column_head
                column_head = df.columns
                data = np.array(df)
            else:
                data = df           
        
                # base case for checking leaf node.
            
            if (self.purity_check(data)):
                return self.classify(data)
            
            # Recursive part of the question.
            else:    
                flag += 1
    
                # finding all splits of data, finding best split and then splitting dataset into two parts.
                
                self.all_splits(data)
                column,value = self.best_split(data)
                left_data, right_data = self.split_data(data,column,value)
                

                feature = column_head[column]
                boolean = "{} <= {}".format(feature, value)
                tree = {boolean: []}
        
                # Yes ans No answer for the boolean question.
                # Recursive call
                yes = self.fit(left_data, flag)
                no = self.fit(right_data, flag)
        
                # Appending the answers in the tree dictionary.
                tree[boolean].append(yes)
                tree[boolean].append(no)
                
                # storing the tree in self.tree.
                self.tree = tree
                return tree
            
        
        def purity_check(self,data):
            
            """ It takes a dataset as argument and returns True if all the target values are same else False. """ 
            
            if len(np.unique(data[:,-1]))==1:
                return True
            else:
                return False
        
        def classify(self,data):
            
            """ It takes a dataset as argument and returns the mode of all classes in target feature. """
            
            classes,counts = np.unique(data[:,-1],return_counts=True)
            max_index = counts.argmax()
            return classes[max_index]
        
        
        def all_splits(self,data):
            
            """ Takes all possible combinatons of splits and appends it into a dictionary."""
        
            self.splits = {}
            for i in range(data.shape[1]-1):
                column = data[:,i]
                unique = np.unique(column)
                split_array = []
                for j in range(len(unique)):
                    if j!=0:
                        split = (unique[j]+ unique[j-1])/2
                        split_array.append(split)
                self.splits[i] = split_array
            

        def split_data(self,data,column,value):
            
            """ splits the dataset into left and right given the column and split value. """
            
            left_data = data[data[:,column] <= value]
            right_data = data[data[:,column]>value]
            return left_data,right_data
        
        
        def entropy(self,data):
            
            """ Calculates the Entropy of the data."""
            
            y=data[:,-1]
            _,counts = np.unique(y,return_counts=True)
            counts = counts/counts.sum()
            return (-counts*np.log2(counts)).sum()
        
        
        def complete_entropy(self,left_data,right_data):
            
            """ Calculates the complete entropy after the split."""
            total = len(left_data) + len(right_data)
            left_prob = (len(left_data)/total)*(self.entropy(left_data))
            right_prob = (len(right_data)/total)*(self.entropy(right_data))        
            return left_prob + right_prob
            
        
        def best_split(self,data):
            
            """ finds the best split i.e the one with lowest entropy is the best split. Returns the best column and split value."""
            
            lowest_entropy = 10000000
            for i in range(len(self.splits)):
                column = i
                for value in self.splits[i]:
                    
                    left_data,right_data = self.split_data(data,column,value)
                    entropy = self.complete_entropy(left_data,right_data)
                    if entropy <= lowest_entropy:
                        lowest_entropy = entropy 
                        best_column = column
                        best_value = value
            return best_column,best_value
        
        
        def predict_example(self,row,tree):
            
            """ Takes a row(one example) and the tree as argument returns the answer for that particular example."""
            
            boolean = list(tree.keys())[0]
            feature,_,value = boolean.split()
            
            if row[feature].iloc[0]<=float(value):
                
                # Yes answer.
                answer = tree[boolean][0]
            else:
                # No answer
                answer = tree[boolean][1]
            
            if not isinstance(answer,dict):
                return answer
            # Recursive Call.
            else:
                return self.predict_example(row,answer)
        
        
        def predict(self,data):
            
            """ Takes test data as argument and returns a numpy array of prediction."""
            
            pred = []
            tree = self.tree
            for i in range(data.shape[0]):
                
                row =  data.iloc[i:(i+1),:]
                pred.append(self.predict_example(row,tree))
                
            return np.array(pred)
        
        
        def accuracy(self,test_df,pred):
            
            test = np.array(test_df.iloc[:,-1])
            sum1=0
            for i in range(len(pred)):
                if test[i]== pred[i]:
                    sum1+=1
            return sum1/len(pred)
        
        
        def train_test_split(self,df, test_size):
        
            if isinstance(test_size, float):
                test_size = round(test_size * len(df))

            indices = df.index.tolist()
            test_indices = random.sample(population=indices, k=test_size)

            test_df = df.loc[test_indices]
            train_df = df.drop(test_indices)
        
            return train_df, test_df
