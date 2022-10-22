import matplotlib.pyplot as plt
import numpy as np

def compare_acc(pred, y):
  num_right=0
  num=0
  for i in range(y.shape[0]):
    if abs(pred[i]-y[i])<0.5:
      num_right+=1
    num+=1
  return num_right/num

def LIME(model, col_info, get_data, seq_length, train_prop):
  importance = {}
  for i in range(len(col_info)-1):
    for j,c in enumerate(col_info):
      if i==j:
        c.InfoType_used = False
      else:
        c.InfoType_used = True
    x_train, y_train, x_test, y_test = get_data(col_info, "moving_two_files_modbus_6RTU", seq_length, train_prop)
    pred = model.predict(x_train)
    temp_train_acc = compare_acc(model.predict(x_train),y_train)
    temp_test_acc = compare_acc(model.predict(x_test),y_test)
    print(f"Testing accuracy with {col_info[i].InfoType_name} set to 0")
    print(f"Accuracy on train: {temp_train_acc}, change in accuracy: {temp_train_acc - train_acc}")
    pred = model.predict(x_test)
    print(f"Accuracy on test: {temp_test_acc}, change in accuracy: {temp_test_acc - test_acc}")
    importance[col_info[i].InfoType_name] = -1*(temp_train_acc - train_acc + temp_test_acc - test_acc)/2.0

  names = list(importance.keys())
  values = list(importance.values())

  plt.bar(range(len(importance)), values, tick_label=names)
  plt.xticks(rotation = 45)
  plt.show()