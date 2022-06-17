import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



def load_dataset(path_to_csv, col_names= []):
  """
  Loads the dataset of our choice from the csv data format file

  Arguments:
  path_to_csv: A string indicating the path to the csv file to be loaded
  col_names: A list to be passed in for getting the column headings in dataframe. 

  returns:
  A pandas dataframe for the dataset
  """
  if col_names == []:
    df = pd.read_csv(path_to_csv)
  else:
    df = pd.read_csv(path_to_csv, names = col_names)
  
  return df





def data_summary(train_set_X, test_set_X, train_set_y, test_set_y):
  m_train = train_set_X.shape[0]
  m_test =  test_set_X.shape[0]
  num_px = train_set_X.shape[1]


  print ("Number of training examples: m_train = " + str(m_train))
  print ("Number of testing examples: m_test = " + str(m_test))
  print ("Height/Width of each image: num_px = " + str(num_px))
  print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
  print ("train_set_x shape: " + str(train_set_X.shape))
  print ("train_set_y shape: " + str(train_set_y.shape))
  print ("test_set_x shape: " + str(test_set_X.shape))
  print ("test_set_y shape: " + str(test_set_y.shape))
