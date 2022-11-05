import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# LSTM has the supervised and unsupervised models.
from LSTM import get_model, get_auto_encoder
from Interpretability import compare_acc

def supervised_learn(x_train, y_train, x_test, y_test, seq_length = 30, train_prop = 0.8, units=100, epochs=5, model=None, verbose=False):
  """
  Creates a basic LSTM which learns to classify data as malicious or not
  based on labeled data. A worse version of Alex's model from a long time 
  ago. 

  Inputs: 
    seq_length: in order to train in parallel, the model is set to stateless and
                fed sequences of fixed length to take advantage of parallel 
                architectures for training. this is the length of those fixed
                sequences. longer is more realistic but will allow for fewer 
                training examples so more likely to overfit. 
    train_prop: proportion of the data to be used for trianing from 0 to 1.0
    units: Number of LSTM units. More units can learn more complicated functions.
           More units also increase the likelihood of overfitting.
    verbose: if True this function will print some information
    x_train: the training data in chunks of sequence length.
    y_train: the labels, a binary classification
    x_test: '' ''
    y_test: '' '' 

  Outputs:
    model: the trained LSTM expecting complete sequences
    history: the keras model history
    train_acc: training accuracy
    test_acc: testing accuracy
  """
  
  if verbose:
    print(f"Sequence data shape: x {x_train.shape}, y {y_train.shape}")

  if model==None:
    model = get_model(num_inputs=x_train.shape[2], units=units, seq_length=seq_length)
  
  if verbose:
    print(f"Ready to train, np array contains nans? x: {np.isnan(x_train).any()}, y: {np.isnan(y_train).any()}")
  history = model.fit(x=x_train,y=y_train,epochs=epochs,validation_data=(x_test, y_test))

  pred = model.predict(x_train)
  if verbose:
    print(pred)
    print(f"Accuracy on train: {compare_acc(pred, y_train)}")
  pred = model.predict(x_test)
  if verbose:
    print(f"Accuracy on test: {compare_acc(pred, y_test)}")

  train_acc = compare_acc(model.predict(x_train),y_train)
  test_acc = compare_acc(model.predict(x_test),y_test)

  return model, history, train_acc, test_acc

def unsupervised_learn(x_train, x_val, x_test, seq_length=10, units=[64,64], epochs=100, model=None):
  """
  This function trains an auto encoder model on training data and 
  also optionally records the loss on validation data and test data.
  The reason for separating validation and test is so that the test
  data may be malicious packets in order to get an idea of the cutoff
  to be used for reconstruction error.

  Inputs: 
    x_train: numpy array of numerical input of shape 
             [n-seq_length, seq_length, num_features]
             This is not a data frame, this is the input
             constructed from the data frame using DataLoader's
             'make_supervised_sequence' function
    x_val:   Same thing but validation data
    x_test:  Same thing but these are sequences ending in 
             malicious packets. 
    seq_length: How long of a sequence should the LSTM learn
                longer sequences may require more recurrent
                units in order to encode properly.
    units: list of length 2 [n_encoder_units, n_decoder_units]
    epochs: How long to train
    model: a pretrained autoencoder LSTM in the case of fine 
           tuning / transfer learning. 

  Ouptuts:
    model: The trained LSTM. 
    train_l, val_l, test_l: the loss recorder at each epoch
                            for train, val, and test data. 
                            We hope to see high test error
  """
  
  print("Now performing unsupervised learning")
  print(f"x train shape: {x_train.shape}, x val shape: {x_val.shape}, x test shape: {x_test.shape} ")
  #input("Hit enter to train")
  if model==None:
    model = get_auto_encoder(num_inputs=x_train.shape[2], units=units, seq_length=seq_length)

  print(f"Ready to train. np array contains nans? x: {np.isnan(x_train).any()}, y: {np.isnan(x_val).any()}")
  #print(x_train)
  #print(np.sum(y_train))
  train_l=[]
  val_l=[]
  test_l=[]
  for i in range(epochs):
    history = model.fit(x=x_train,y=x_train,epochs=1,validation_data=(x_val, x_val))
    train_l.append(history.history["loss"][0])
    val_l.append(history.history["val_loss"][0])
    test_l.append(model.evaluate(x_test, x_test))
  
  return model, train_l, val_l, test_l
