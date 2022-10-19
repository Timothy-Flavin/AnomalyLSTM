import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Get the data as a pandas dataframe
def read_as_data_frame(fileName, verbose=False):
  data = pd.read_csv("Data/"+fileName+".csv",index_col=0).fillna(0)
  #data = data.astype("float32")
  if verbose:
    print(data.head())
  return data

class InfoType:
  InfoType_name=""
  InfoType_length=1
  InfoType_used=True
  InfoType_transform=None
  
  def default_transform(self,col):
    #print(f"Default transform col: {col}")
    return np.reshape(np.array(col),(len(col),1))

  def __init__(self, name, i_len=1, used=True, transform=None):
    self.InfoType_name=name
    self.InfoType_length=i_len
    self.InfoType_used=used
    if transform is None:
      self.InfoType_transform = self.default_transform
    else:
      self.InfoType_transform = transform

  def __len__(self):
    return self.InfoType_length
  def transform(self, col):
    return self.InfoType_transform(col)
# col is a column of the data frame
# sep is the separator character, like ':' in ethernet addresses
# n is the number of parts (n is not decided dynamically in case data is bad 
def address_to_numpy(col, sep, n, base):
  arr=np.zeros((len(col),n), dtype="float32")
  for i,r in enumerate(col):
    if r==0:
      continue
    #print(r)
    arr[i] = np.array([(lambda x: int(x,base))(x) for x in r.split(sep)])
  return arr

def eth_to_numpy(col):
  return address_to_numpy(col, ":", 6, 16)
def ip_to_numpy(col):
  return address_to_numpy(col, ".", 4, 10)

def join_columns(df, data_types):
  c_sum = 0
  data = np.zeros((len(df.index), sum(len(x) for x in data_types)))
  #print(f"Joining columns, data shape: {data.shape}")
  for c in data_types:
    i=c_sum
    temp_arr = c.transform(df[c.InfoType_name])
    while i<c_sum+len(c):
      #print(f"c_sum: {c_sum}, i: {i}, i-c_sum: {i-c_sum}, tempshape: {temp_arr.shape}")
      if c.InfoType_used:
        data[:,i] = temp_arr[:,i-c_sum]
      else:
        colavg = np.full((data.shape[0]),np.mean(temp_arr,axis=0)[i-c_sum])
        #print(f"Col {c.InfoType_name} not used. data[:,{i}] would be {temp_arr[:,i-c_sum]}, but it is instead the average: {colavg}")
        data[:,i] = colavg
        #input("Press enter to continue")
      i+=1
    c_sum+=len(c)
  return data


def normalize(data):
  scal = StandardScaler()
  scal.fit_transform(data)
  return scal, data
# Taken from https://medium.com/mlearning-ai/lstm-with-keras-data-reshaping-in-many-to-one-architecture-c7d3669e3a5c
# No reason to reinvent the wheel on reshaping data
def make_sequence(data, seq_len):
  '''
  input:
    data - the numpy matrix of (n, p+1) shape, where n is the number of rows,
            p+1 is the number of predictors + 1 target column
    k    - the length of the sequence, namely, the number of previous rows 
            (including current) we want to use to predict the target.
  output:
    X_data - the predictors numpy matrix of (n-k, k, p) shape
    y_data - the target numpy array of (n-k, 1) shape
  '''
  # initialize zero matrix of (n-k, k, p) shape to store the n-k number
  # of sequences of k-length and zero array of (n-k, 1) to store targets
  X_data = np.zeros([data.shape[0]-seq_len, seq_len, data.shape[1]-1])
  y_data = []
  
  # run loop to slice k-number of previous rows as 1 sequence to predict
  # 1 target and save them to X_data matrix and y_data list
  for i in range(seq_len, data.shape[0]):
    cur_sequence = data[i-seq_len: i, :-1]
    cur_target = data[i-1, -1]
    
    X_data[i-seq_len,:, :] = cur_sequence.reshape(1, seq_len, X_data.shape[2])
    y_data.append(cur_target)
  
  return X_data, np.asarray(y_data)

