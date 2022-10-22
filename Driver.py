import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataLoader import InfoType, modbus_to_data_frame
from DataLoader import ip_to_numpy, eth_to_numpy, join_columns
from DataLoader import make_supervised_sequence
from DataLoader import normalize
from Interpretability import compare_acc, LIME
from LSTM import get_model, get_auto_encoder
from PrinterDataLoader import read_printer_as_df, max_normalize, protocols_to_dummy
import tensorflow as tf
import pickle as pk

# Setup for moving two files modbus 6rtu data
moving_two_files_cols = [
  InfoType("len"),
  InfoType("ethernet_dst",i_len=6, used=True, transform=eth_to_numpy),
  InfoType("ethernet_src",i_len=6, used=True, transform=eth_to_numpy),
  InfoType("ethernet_ipv4_src",i_len=4, used=True, transform=ip_to_numpy),
  InfoType("ethernet_ipv4_dst",i_len=4, used=True, transform=ip_to_numpy),
  InfoType("ethernet_ipv4_tcp_srcport"),
  InfoType("ethernet_ipv4_tcp_dstport"),
  InfoType("label")
]

# Setup for printer conncting unknown IP data
#"len","ethernet_dst","ethernet_src","ipv4_dst","ipv4_src","protocol","version"
printer_cols = [
  InfoType("len"),
  InfoType("ethernet_dst",i_len=6, used=True, transform=eth_to_numpy),
  InfoType("ethernet_src",i_len=6, used=True, transform=eth_to_numpy),
  InfoType("ipv4_dst",i_len=4, used=True, transform=ip_to_numpy),
  InfoType("ipv4_src",i_len=4, used=True, transform=ip_to_numpy),
  InfoType("protocol",i_len=12, used=True, transform=protocols_to_dummy),
  InfoType("version"),
  InfoType("label"),
]

def get_modbus_data(cols, filename, seq_length, train_prop, subset_by_label=-1, verbose=False):
  """
  subset_by_label means that we will get only the data with the desired label
  This is so we can test the unsupervised learner by training it to encode and
  decode normal data and then compare it's ability to decode with that of the 
  anomolous data. This way we can give a model normal real world data with some
  idea of whether it will detect anomolies 
  """
  data_frame = modbus_to_data_frame(filename)

  if subset_by_label == 1:
    data_frame = data_frame.loc[data_frame.iloc[:,-1] == 1]
    if verbose:
      print("Getting only data with label 1")
      print(data_frame.head)
  if subset_by_label == 0:
    data_frame = data_frame.loc[data_frame.iloc[:,-1] == 0]
    if verbose:
      print("Getting only data with label 0")
      print(data_frame.head)

  data = join_columns(data_frame,cols)
  #print(f"Printing modbus data: {data}")
  scal, data = normalize(data)
  seq_data, seq_labels = make_supervised_sequence(data, seq_length)

  x_train = seq_data[:int(train_prop*seq_data.shape[0]),:,:]
  x_test = seq_data[int(train_prop*seq_data.shape[0]):,:,:]
  y_train = seq_labels[:int(train_prop*seq_labels.shape[0]),]
  y_test = seq_labels[int(train_prop*seq_labels.shape[0]):,]

  return x_train, y_train, x_test, y_test

def get_printer_data(cols, seq_length, train_prop, subset_by_label=-1, verbose=False):
  data_frame, skipped = read_printer_as_df()

  if subset_by_label == 1:
    data_frame = data_frame.loc[data_frame.iloc[:,-1] == 1]
    if verbose:
      print("Getting only data with label 1")
      print(data_frame.head)
  if subset_by_label == 0:
    data_frame = data_frame.loc[data_frame.iloc[:,-1] == 0]
    if verbose:
      print("Getting only data with label 0")
      print(data_frame.head)

  data = join_columns(data_frame,cols)
  #print(f"Printing modbus data: {data}")
  data,maxes = max_normalize(data)
  print(data.shape)
  print(data[0:10,0:10])
  seq_data, seq_labels = make_supervised_sequence(data, seq_length)
  print(seq_data.shape)
  print(seq_data[0,0:10,0:10])
  input("Is this the right shape?")

  x_train = seq_data[:int(train_prop*seq_data.shape[0]),:,:]
  x_test = seq_data[int(train_prop*seq_data.shape[0]):,:,:]
  y_train = seq_labels[:int(train_prop*seq_labels.shape[0]),]
  y_test = seq_labels[int(train_prop*seq_labels.shape[0]):,]

  return x_train, y_train, x_test, y_test

def supervised_learn(seq_length = 30, train_prop = 0.8, units=100, verbose=False):
  """
  Creates a bastic LSTM which learns to classify data as malicious or not
  based on labeled data. 

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
  
  x_train, y_train, x_test, y_test = get_modbus_data(moving_two_files_cols, "moving_two_files_modbus_6RTU", seq_length, train_prop)
  
  if verbose:
    print(f"Sequence data shape: x {x_train.shape}, y {y_train.shape}")
  model = get_model(num_inputs=x_train.shape[2], units=units, seq_length=seq_length)
  
  if verbose:
    print(f"Ready to train, np array contains nans? x: {np.isnan(x_train).any()}, y: {np.isnan(y_train).any()}")
  history = model.fit(x=x_train,y=y_train,epochs=5,validation_data=(x_test, y_test))

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




def unsupervised_learn(seq_length=10, train_prop=0.8):
  print("Now performing unsupervised learning")
  # We can ignore the labels here because we are training 
  # the model to recreate x
  x_train, _1, x_val, _2 = get_printer_data(cols=printer_cols, seq_length=seq_length, train_prop=train_prop, subset_by_label=0)
  x_test, _1, _2, _3 = get_printer_data(cols=printer_cols, seq_length=seq_length, train_prop=1, subset_by_label=1)

  print(f"x train shape: {x_train.shape}, x val shape: {x_val.shape}, x test shape: {x_test.shape} ")
  #input("Hit enter to train")
  model = get_auto_encoder(num_inputs=x_train.shape[2], units=[128,64], seq_length=seq_length)

  print(f"Ready to train, np array contains nans? x: {np.isnan(x_train).any()}, y: {np.isnan(x_val).any()}")
  #print(x_train)
  #print(np.sum(y_train))
  train_l=[]
  val_l=[]
  test_l=[]
  for i in range(100):
    history = model.fit(x=x_train,y=x_train,epochs=1,validation_data=(x_val, x_val))
    train_l.append(history.history["loss"][0])
    val_l.append(history.history["val_loss"][0])
    test_l.append(model.evaluate(x_test, x_test))
  print(train_l)
  print(val_l)
  print(test_l)
  plt.plot(train_l)
  plt.plot(val_l)
  plt.plot(test_l)
  plt.legend(["Train loss", "Validation loss", "Test Loss"])
  plt.xlabel("Epoch")
  plt.ylabel("Mean absolute Error")
  plt.title("Comparing performance on anomolous data to regular data")
  plt.show()
  print("Done")
  #train_errors = tf.keras.losses.mean_absolute_error(x_train, model.predict(x_train))
  #print(train_errors)
  #input("I know the shape now ")
  train_errors = tf.keras.losses.mean_absolute_error(x_train, model.predict(x_train))[:,-1]
  test_errors = tf.keras.losses.mean_absolute_error(x_test, model.predict(x_test))[:,-1]

  

  #print(x_train[0,:,:])
  #print(model.predict(np.reshape(x_train[0,:seq_length,:],(1,x_train.shape[1],x_train.shape[2]))))
  #input("???")

  l_mean = np.mean(train_errors)
  l_std = np.std(train_errors)

  print(f"train error shape {train_errors.shape}, test error shape: {test_errors.shape}")
  for k in range(0,12):
    i = 1+k/4.0
    print(f"Cutoff {i} \nTrue Negative: {train_errors[train_errors<l_mean+i*l_std].shape} ")
    print(f"True Positives: {test_errors[test_errors>l_mean+i*l_std].shape}")
    print(f"False Positives: {train_errors[train_errors>l_mean+i*l_std].shape}")
    print(f"False Negatives: {test_errors[test_errors<l_mean+i*l_std].shape}")
  input("'enter' to see some test results and a graph...")

  train_w = np.empty(train_errors.shape)
  train_w.fill(1/train_errors.shape[0])
  test_w = np.empty(test_errors.shape)
  test_w.fill(1/test_errors.shape[0])
  #plt.hist(train_errors.numpy(), bins=30, color="blue")
  #plt.hist(test_errors.numpy(), color="red")
  plt.hist([train_errors.numpy(), test_errors.numpy()], bins=40, weights=[train_w, test_w], label=['Train', 'Test'])
  plt.legend(["Normal Data","Unknown: IP 169.x.x.x"])
  plt.title("Histogram of Reconstruction Errors")
  plt.xlabel("MAE of Reconstructed Packets")
  plt.ylabel("Number of Occurences (Scaled)")
  plt.show()

  print(f"Train Loss mean: {l_mean}, Train Loss stdv: {l_std}")
  print(f"Test Loss mean: {np.mean(test_errors)}, Test Loss stdv: {np.std(test_errors)}")

  mae1 = model.evaluate(x_test,x_test)
  mae2 = tf.keras.losses.mean_absolute_error(x_test, model.predict(x_test))
  print(mae2[:,-1])
  print([mae1,np.mean(mae2)])

  return model, mae1, x_train, x_test

model, mae, train_data, test_data = unsupervised_learn()
