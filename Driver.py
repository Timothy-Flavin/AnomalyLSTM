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
from Interpretability import compare_acc, LIME, unsupervised_results, Shapely

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
  InfoType("ethernet_ipv4_tcp_srcport", used=True),
  InfoType("ethernet_ipv4_tcp_dstport", used=True),
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
  InfoType("version", used=True),
  InfoType("label"),
]

# overwriting the other definition to do transfer learning by making
# the columns be the same as the printer data
moving_two_files_cols = [
  InfoType("len"),
  InfoType("ethernet_dst",i_len=6, used=True, transform=eth_to_numpy),
  InfoType("ethernet_src",i_len=6, used=True, transform=eth_to_numpy),
  InfoType("ethernet_ipv4_src",i_len=4, used=True, transform=ip_to_numpy),
  InfoType("ethernet_ipv4_dst",i_len=4, used=True, transform=ip_to_numpy),
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
  InfoType("label"),
]


def printer_unsupervised(model=None, train_l=None, val_l=None, test_l=None, lime=False, shapely=False):
  seq_length=10
  train_prop=0.8
  train_x, train_y, val_x, val_y, train_df, maxes = get_printer_data(printer_cols, "Data/PrinterPackets.csv", seq_length, train_prop=train_prop, subset_by_label=0, whole_seq_bad=False, verbose=False)
  #test_x, _3, _4, _5, df_test, maxes = get_printer_data(printer_cols, "Data/PrinterPackets.csv", seq_length, train_prop=1.0, subset_by_label=1, maxes=maxes, verbose=False)
  print(f"subset by label =0: x shape train: {train_x.shape}, val: {val_x.shape}")
  test_x, test_y, _, _2, test_df, maxes = get_printer_data(printer_cols, "Data/PrinterPackets.csv", seq_length, train_prop=train_prop, subset_by_label=1, whole_seq_bad=False, maxes=maxes, verbose=False)

  #test_x, _3, _4, _5, df_test, maxes = get_printer_data(printer_cols, "Data/PrinterPackets.csv", seq_length, train_prop=1.0, subset_by_label=1, maxes=maxes, verbose=False)
  print(f"subset by label =1: x shape train: {test_x.shape}")

  model, train_l, val_l, test_l = unsupervised_learn(train_x,val_x,test_x, seq_length, [64,64], 100, model=model)
  unsupervised_results(model, train_x, test_x, train_l, val_l, test_l)
  if lime:
    LIME(model, printer_cols, get_printer_data, seq_length, train_prop, file_name="Data/PrinterPackets.csv", supervised=False)
  if shapely:
    Shapely(model, printer_cols, get_printer_data, seq_length, train_prop, file_name="Data/PrinterPackets.csv", supervised=False, n_samps=10)
  # for supervised, file_name = "moving_two_files_modbus_6RTU"
  return model


def modbus_unsupervised(model=None, lime=False, shapely=False, verbose=False):
  """
  This is basically an example for training a model and showing the results.
  """
  seq_length=10
  train_prop=0.8
  print("Modbus data:")
  # Getting the non-malicious sequences
  train_x, train_y, val_x, val_y, train_df, maxes = get_modbus_data(moving_two_files_cols, "moving_two_files_modbus_6RTU", seq_length, train_prop=train_prop, subset_by_label=0, whole_seq_bad=False, verbose=True)
  print(f"subset by label =0: x shape train: {train_x.shape}, val: {val_x.shape}")

  # getting the sequences ending in a malicious packet
  test_x, test_y, _, _2, test_df, maxes = get_modbus_data(moving_two_files_cols, "moving_two_files_modbus_6RTU", seq_length, train_prop=1.0, subset_by_label=1, maxes=maxes, whole_seq_bad=False, verbose=True)
  print(f"subset by label =1: x shape train: {test_x.shape}")


  model, train_l, val_l, test_l = unsupervised_learn(train_x,val_x,test_x, seq_length, [64,64], 100, model=model)
  unsupervised_results(model, train_x, test_x, train_l, val_l, test_l)
  if lime:
    LIME(model, printer_cols, get_modbus_data, seq_length, train_prop, file_name="moving_two_files_modbus_6RTU", supervised=False)
  if shapely:
    Shapely(model, printer_cols, get_modbus_data, seq_length, train_prop, file_name="moving_two_files_modbus_6RTU", supervised=False, n_samps=10)
  # for supervised, file_name = "moving_two_files_modbus_6RTU"

# Transfer learning: 
model = modbus_unsupervised()
model = printer_unsupervised(lime=True, model=model)

# No Transfer learning: 
model = printer_unsupervised(lime=True, model=None)