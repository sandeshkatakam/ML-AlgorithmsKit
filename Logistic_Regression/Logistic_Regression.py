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


def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """

    ### START CODE HERE ### (≈ 1 line of code)
    s = 1/(1+np.exp(-z))
    ### END CODE HERE ###
    
    return s

def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    w = np.zeros((dim,1))
    b = 0

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b

def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    """
    
    m = X.shape[1]
    
    # FORWARD PROPAGATION (FROM X TO COST)
    A = sigmoid(np.dot(w.T,X) + b)              # compute activation
    cost = np.sum(((- np.log(A))*Y + (-np.log(1-A))*(1-Y)))/m  # compute cost

    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = (np.dot(X,(A-Y).T))/m
    db = (np.sum(A-Y))/m

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost


if __name__ == "__main__":
    # Some tests :
    print ("sigmoid([0, 2]) = " + str(sigmoid(np.array([0,2]))))
    dim = 2
    w, b = initialize_with_zeros(dim)
    print ("w = " + str(w))
    print ("b = " + str(b))
