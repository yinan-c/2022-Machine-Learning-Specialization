# UNQ_C2
# GRADED FUNCTION: my_dense
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from autils import *

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
def sigmoid(z):
    """
    Computes sigmoid of z
    Args:
      z (ndarray): input to sigmoid function
    Returns:
      ndarray: sigmoid of z
    """
    return 1.0/(1.0+np.exp(-z))
def my_dense(a_in, W, b, g):
    """
    Computes dense layer
    Args:
      a_in (ndarray (n, )) : Data, 1 example 
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (j, )) : bias vector, j units  
      g    activation function (e.g. sigmoid, relu..)
    Returns
      a_out (ndarray (j,))  : j units
    """
    units = W.shape[1]
    a_out = np.zeros(units)
### START CODE HERE ### 
    #a_in = a_in.reshape(1)
    #b = b.reshape(1)
    # ! You don't  have to reshape an array of (m,) to (1,m), this is how broadcasting works
    z_j_array = np.matmul(a_in,W) + b # This will be a mat of 1 x j
    a_out = sigmoid(z_j_array) 
### END CODE HERE ### 
    return(a_out)

# Quick Check
x_tst = 0.1*np.arange(1,3,1)#.reshape(2,)  # (1 examples, 2 features)
print(x_tst)
x_tst = x_tst.reshape(2,)
print(x_tst)
W_tst = 0.1*np.arange(1,7,1).reshape(2,3) # (2 input features, 3 output features)
b_tst = 0.1*np.arange(1,4,1).reshape(3,)  # (3 features)
A_tst = my_dense(x_tst, W_tst, b_tst, sigmoid)#.numpy()
print(A_tst)
