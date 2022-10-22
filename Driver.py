import numpy as np
# Data loader gets data from a csv file and turns
# it into something the model can use whether that 
# be leo fernandez data or printer data.
from DataLoader import InfoType
from DataLoader import ip_to_numpy, eth_to_numpy, protocols_to_dummy
from DataLoader import get_modbus_data, get_printer_data

# Interpretability shows the results of the models 
# in addition to some basic analysis on which features
# lead to certain results. 
from Interpretability import compare_acc, LIME, unsupervised_results

# Learn trains a model on data and returns the model 
# and scores
from Learn import unsupervised_learn, supervised_learn

# Setup for moving two files modbus 6rtu data, These
# are the columns of interest encoded as InfoTypes 
# so that they can be used generally in the data
# loading functions
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



# We can ignore the labels here because we are training 
# the model to recreate x
seq_length=10
train_prop=0.8
train_x, train_y, val_x, val_y, df, maxes = get_printer_data(printer_cols, "Data/PrinterPackets.csv", seq_length, train_prop=train_prop, subset_by_label=-1, whole_seq_bad=True, verbose=False)
#test_x, _3, _4, _5, df_test, maxes = get_printer_data(printer_cols, "Data/PrinterPackets.csv", seq_length, train_prop=1.0, subset_by_label=1, maxes=maxes, verbose=False)
# need to remove sequences with label = 1 from train
# and val because those are the unknown IPs and will 
# used to see how the model does on bad packets. 
badbois=np.where(train_y>0)[0]
badbois2 = np.where(val_y>0)[0] 
for i in range(len(badbois)):
  print(i)
  print(badbois[i])
print(df.iloc[badbois+seq_length-1].head(50))
print(df.iloc[badbois2+seq_length-1 + train_y.shape[0] + seq_length])
input("move on?")
model, train_l, val_l, test_l = unsupervised_learn(train_x,val_x,test_x, seq_length, [64,64], 100)
unsupervised_results(model, train_x, test_x, train_l, val_l, test_l)
LIME(model, printer_cols, get_printer_data, seq_length, train_prop, file_name="")

# for supervised, file_name = "moving_two_files_modbus_6RTU"