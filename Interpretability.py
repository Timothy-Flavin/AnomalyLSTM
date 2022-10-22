import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

def compare_acc(pred, y):
  num_right=0
  num=0
  for i in range(y.shape[0]):
    if abs(pred[i]-y[i])<0.5:
      num_right+=1
    num+=1
  return num_right/num

def LIME(model, col_info, get_data, seq_length, train_prop, file_name):
  importance = {}

  # getting accuracy with all features as a baseline
  x_train, y_train, x_test, y_test, df, maxes = get_data(col_info, file_name, seq_length, train_prop)
  train_acc = compare_acc(model.predict(x_train),y_train)
  test_acc = compare_acc(model.predict(x_test),y_test)

  for i in range(len(col_info)-1):
    for j,c in enumerate(col_info):
      if i==j:
        c.InfoType_used = False
      else:
        c.InfoType_used = True
    x_train, y_train, x_test, y_test, tdf, tmaxes = get_data(col_info, file_name, seq_length, train_prop)
    temp_train_acc = compare_acc(model.predict(x_train),y_train)
    temp_test_acc = compare_acc(model.predict(x_test),y_test)
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
  #print(mae2[:,-1])
  #print([mae1,np.mean(mae2)])