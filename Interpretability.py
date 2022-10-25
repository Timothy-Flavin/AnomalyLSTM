import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import random

def get_errors(model, data):
  """
  Given an autoencoder model and data, returns the 
  list of errors for reconstructing last packet in 
  the sequence in addition to the mean of those 
  errors and the standard deviation to be use in GMM
  """
  errors = tf.keras.losses.mean_absolute_error(data, model.predict(data))[:,-1]
  l_mean = np.mean(errors)
  l_std = np.std(errors)
  return errors, l_mean, l_std

def compare_acc(pred, y):
  """
  For the supervised model, this returns the accuracy from 0 to 1
  Inputs: 
    pred: the list of predictions (0 or 1)
    y: the list of labels (0 or 1)

  Outputs:
    accuracy from 0.0 to 1.0
  """
  num_right=0
  num=0
  for i in range(y.shape[0]):
    if abs(pred[i]-y[i])<0.5:
      num_right+=1
    num+=1
  return num_right/num

def compare_acc_unsupervised(errors, mean, deviation, labels):
  """
  Inputs: 
    errors: A list of reconstruction errors for the last packet of 
            each sequence. Can be obtained using 'get_errors(model, data)'
    mean: the mean of those errors. Also obtained with 
          'get_errors(model, data)'
    deviation: tunable parameter equal to standard deviation*n. This is
               how sensitive you want the model to be to anomolies 
    labels: tells is whether that packet should have a high or low
            reconstruction error.  

  Outputs:
    accuracy from 0 to 1
  """
  num_right=0
  num=0
  for i in range(errors.shape[0]):
    if (abs(errors[i]-mean)<deviation and labels[i]==0) or (abs(errors[i]-mean)>deviation and labels[i]==1):
      num_right+=1
    num+=1
  return num_right/num

def LIME(model, col_info, get_data, seq_length, train_prop, file_name, supervised=True, n_stdv=2):
  importance = {}

  # getting accuracy with all features as a baseline
  #printer_cols, "Data/PrinterPackets.csv", seq_length, train_prop=train_prop, subset_by_label=-1, whole_seq_bad=True, verbose=False
  x_train, y_train, x_test, y_test, df, maxes = get_data(col_info, file_name, seq_length, train_prop)
  
  print(f"x shape in LIME train: {x_train.shape}, val: {x_test.shape}")
  
  train_acc=0
  test_acc=0
  errors=0
  mean=0
  stdv=0
  if supervised:
    train_acc = compare_acc(model.predict(x_train),y_train)
    test_acc = compare_acc(model.predict(x_test),y_test)
  else:
    print("unsupervised LIME")
    errors, mean, stdv = get_errors(model, x_train)
    test_errors, _1, _2 = get_errors(model, x_test)
    train_acc = compare_acc_unsupervised(errors, mean, stdv*n_stdv,y_train)
    test_acc = compare_acc_unsupervised(test_errors, mean, stdv*n_stdv,y_test)
    print(f"train accuracy: {train_acc}, test: {test_acc}")

  for i in range(len(col_info)-1):
    for j,c in enumerate(col_info):
      if i==j:
        c.InfoType_used = False
      else:
        c.InfoType_used = True
    x_train, y_train, x_test, y_test, tdf, tmaxes = get_data(col_info, file_name, seq_length, train_prop)
    temp_train_acc=0
    temp_test_acc=0
    if supervised:
      temp_train_acc = compare_acc(model.predict(x_train),y_train)
      temp_test_acc = compare_acc(model.predict(x_test),y_test)
    else:
      print("unsupervised LIME")
      temp_errors, temp_mean, temp_stdv = get_errors(model, x_train)
      temp_test_errors, _1, _2 = get_errors(model, x_test)
      temp_train_acc = compare_acc_unsupervised(temp_errors, mean, stdv*n_stdv,y_train)
      temp_test_acc = compare_acc_unsupervised(temp_test_errors, mean, stdv*n_stdv,y_test)
      print(f"temp train accuracy: {train_acc}, test: {test_acc}")
    print(f"Testing accuracy with {col_info[i].InfoType_name} set to 0")
    print(f"Accuracy on train: {temp_train_acc}, change in accuracy: {temp_train_acc - train_acc}")
    print(f"Accuracy on test: {temp_test_acc}, change in accuracy: {temp_test_acc - test_acc}")
    importance[col_info[i].InfoType_name] = -1*(temp_train_acc - train_acc + temp_test_acc - test_acc)/2.0

  names = list(importance.keys())
  values = list(importance.values())

  plt.bar(range(len(importance)), values, tick_label=names)
  plt.xticks(rotation = 45)
  plt.show()


def unsupervised_results(model, x_train, x_test, train_l, val_l, test_l):
  #print(train_l)
  #print(val_l)
  #print(test_l)
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
  
  test_errors, _1, _2 = get_errors(model, x_test)
  train_errors, l_mean, l_std = get_errors(model, x_train)

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
  #print(mae2[:,-1])
  #print([mae1,np.mean(mae2)])


def Shapely(model, col_info, get_data, seq_length, train_prop, file_name, supervised=True, n_stdv=2, n_samps = 1000):
  importance = {}

  ox_train, oy_train, ox_test, oy_test, df, maxes = get_data(col_info, file_name, seq_length, train_prop)
  for c in col_info:
    c.InfoType_used = False
  # getting accuracy with all features as a baseline
  #printer_cols, "Data/PrinterPackets.csv", seq_length, train_prop=train_prop, subset_by_label=-1, whole_seq_bad=True, verbose=False
  x_train, y_train, x_test, y_test, df, maxes = get_data(col_info, file_name, seq_length, train_prop)
  print(f"x shape in Shapely train: {x_train.shape}, val: {x_test.shape}")
  train_acc=0
  test_acc=0
  errors=0
  mean=0
  stdv=0
  if supervised:
    o_train_acc = compare_acc(model.predict(x_train),y_train)
    o_test_acc = compare_acc(model.predict(x_test),y_test)
  else:
    errors, mean, stdv = get_errors(model, x_train)
    best_error, mean, stdv = get_errors(model, ox_train)
    test_errors, _1, _2 = get_errors(model, x_test)
    o_train_acc = compare_acc_unsupervised(errors, mean, stdv*n_stdv,y_train)
    o_test_acc = compare_acc_unsupervised(test_errors, mean, stdv*n_stdv,y_test)

  for n in range(n_samps):
    print(importance)
    
    # create a random ordering of the features and set them all to false
    train_acc = o_train_acc
    test_acc - o_test_acc
    features_permute = list(range(0,len(col_info)))
    random.shuffle(features_permute)
    print(features_permute)
    input(f"hit enter to increment shapely value {n}")
    for c in col_info:
      c.InfoType_used = False
    # we always want the label to be true  
    col_info[-1].InfoType_used = True
    for i in features_permute:
      # skip the label attribute
      if i==len(col_info)-1:
        continue
      for j,c in enumerate(col_info):
        if i==j:
          c.InfoType_used = True
      x_train, y_train, x_test, y_test, tdf, tmaxes = get_data(col_info, file_name, seq_length, train_prop)
      temp_train_acc=0
      temp_test_acc=0
      if supervised:
        temp_train_acc = compare_acc(model.predict(x_train),y_train)
        temp_test_acc = compare_acc(model.predict(x_test),y_test)
      else:
        temp_errors, temp_mean, temp_stdv = get_errors(model, x_train)
        temp_test_errors, _1, _2 = get_errors(model, x_test)
        temp_train_acc = compare_acc_unsupervised(temp_errors, mean, stdv*n_stdv,y_train)
        temp_test_acc = compare_acc_unsupervised(temp_test_errors, mean, stdv*n_stdv,y_test)
      for c in col_info:
        print(f"name: {c.InfoType_name}, used: {c.InfoType_used}")
      print(f"Testing accuracy with {col_info[i].InfoType_name} included")
      print(f"Accuracy on train: {temp_train_acc}, change in accuracy: {temp_train_acc - train_acc}")
      print(f"Accuracy on test: {temp_test_acc}, change in accuracy: {temp_test_acc - test_acc}")
      importance[col_info[i].InfoType_name] = importance.get(col_info[i].InfoType_name,0)+(temp_train_acc - train_acc + temp_test_acc - test_acc)/2.0
      train_acc = temp_train_acc
      test_acc = temp_test_acc
  names = list(importance.keys())
  values = list(importance.values())

  plt.bar(range(len(importance)), values, tick_label=names)
  plt.xticks(rotation = 45)
  plt.show()