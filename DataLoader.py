import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Get the data as a pandas dataframe
def modbus_to_data_frame(fileName, verbose=False):
  data = pd.read_csv("Data/"+fileName+".csv",index_col=0).fillna(0)
  #data = data.astype("float32")
  if verbose:
    print(data.head())
  return data

  
class InfoType:
  """
    Info Type is a wrapper class that I made to support
    The addition or removal of of different types of data
    from the model. It informs functions that convert the
    data from a dataframe to a numpy array

    Members: 
      InfoType_name -> the name of this data such as 
                       'ethernet_dst' or 'len'. if data
                       is read as a dataframe, this name
                       should match the column
      InfoType_length -> the length of this data based on 
                         the way it will be encoded for the
                         model. so ip to numpy saves a value
                         for each subnet so length=4. The 
                         default is 1. 
      InfoType_used -> When used in 'join_columns', setting
                       This value to False will set all entries
                       in a column to equal the column average.
                       This makes this column essentially contain
                       no information at all so that columns can 
                       be removed from a model without retraining.
                       Also usefull in calculating LIME and Shapely
      InfoType_transform -> A function that transforms a pandas df
                            column into a numpy array. For example
                            IP ["1.1.1.1","2.2.2.2"] to 
                            arr [[1,1,1,1], [2,2,2,2]]. The default
                            simply casts the column as a numpy array
                            with one column
  """
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

def address_to_numpy(col, sep, n, base):
  """
  Inputs: 
    col -> column name to be converted such as 'src_addr'
    sep -> the separator character, such as ':' in ethernet addresses
    n -> the number of parts (n is not decided dynamically in case
         of bad data
    base -> base 10 or 16 for IP or Ethernet 
  """
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
  """
  Input: data_types -> a list of 'InfoType's
         df -> a dataframe with columns corrisponding to 
               'InfoType_name's from the 'data_types'
  This function converts 'df' to a numpy array where some 
  of the 'df' columns are a list of values such as ethernet
  or IP addresses. All of the columns need their lists to be
  concatinated into a numpy array so 
  
  instead of a data frame:
  {col1: [192,168,1,1], col2: [ff,ff,ff,ff,ff,ff]}

  We get:
  np.array([192,168,1,1,ff,ff,ff,ff,ff,ff])

  In reality, these should already be all numeric instead of
  the ethernet still being hex strings, but that was an example.
  """
  c_sum = 0
  data = np.zeros((len(df.index), sum(len(x) for x in data_types)))
  print(f"Joining columns, data shape: {data.shape}")
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
  """
  Normalizing the data with sklearn's Standard Scalar
  """
  scal = StandardScaler()
  scal.fit_transform(data)
  return scal, data
# Taken from https://medium.com/mlearning-ai/lstm-with-keras-data-reshaping-in-many-to-one-architecture-c7d3669e3a5c
# No reason to reinvent the wheel on reshaping data
def make_supervised_sequence(data, seq_len):
  '''
  Makes data with labels for malicious or not
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

def make_unsupervised_sequence(data, seq_len):
  '''
  Makes data without labels for unsupervise learning
  input:
    data - the numpy matrix of (n, p) shape, where n is the number of rows,
            p is the number of features
    k    - the length of the sequence, namely, the number of previous rows 
            (including current) we want to use to predict the target.
  output:
    X_data - the predictors numpy matrix of (n-k, k, p) shape
  '''
  # initialize zero matrix of (n-k, k, p) shape to store the n-k number
  # of sequences of k-length and zero array of (n-k, 1) to store targets
  X_data = np.zeros([data.shape[0]-seq_len, seq_len, data.shape[1]])
  
  # run loop to slice k-number of previous rows as 1 sequence to predict
  # 1 target and save them to X_data matrix and y_data list
  for i in range(seq_len, data.shape[0]):
    cur_sequence = data[i-seq_len: i, :]
    X_data[i-seq_len,:, :] = cur_sequence.reshape(1, seq_len, X_data.shape[2])
  return X_data

