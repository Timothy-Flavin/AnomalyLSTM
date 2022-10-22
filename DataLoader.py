import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from operator import indexOf

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


# Version is either 0 for ipv4 or 1 for ipv6
def read_printer_as_df(path="Data/PrinterPackets.csv"):
  """
  Reads the printer data to a csv 
  """
  raw_table = pd.read_csv(path).fillna('')
  clean_table = pd.DataFrame(columns=["len","ethernet_dst","ethernet_src","ipv4_dst","ipv4_src","protocol","version","label"])
  skipped=[]
  for i,row in raw_table.iterrows():
    #print(row)
    rowdict={}
    rowdict["len"] = row["Length"]
    rowdict["protocol"] = row["Protocol"]

    #version 0 for ip4 1 for ip6
    if row["Version"] == '':
      rowdict['version']=0
    elif int(row["Version"]) == 6:
      rowdict['version']=1
    else:
      print("Up oh")
      rowdict['version']=-1

    #Get ethernet source and destination. if null, skip this packet
    if row["ethernet_dst"] != '':
      rowdict["ethernet_dst"] = row["ethernet_dst"]
    else:
      Warning("Missing ethernet dst")
      skipped.append({"reason":"no_ethernet_dst", "index": i})
      continue
    if row["ethernet_src"] != '':
      rowdict["ethernet_src"] = row["ethernet_src"]
    else:
      Warning("Missing ethernet src")
      skipped.append({"reason":"no_ethernet_src", "index": i})
      continue

    
    # Getting ipv4 src 
    if row["arp_ipv4_src"] != '':
      rowdict["ipv4_src"] = row["arp_ipv4_src"]
    elif row["ethernet_ipv4_src"] != '':
      rowdict["ipv4_src"] = row["ethernet_ipv4_src"]
    elif row['Version'] != '':
      rowdict["ipv4_src"] = "-1.-1.-1.-1"
    else:
      rowdict["ipv4_src"] = "-255.-255.-255.-255"
      Warning("No ipv4 src found and version was '' instead of '6'. setting ipv4 to -255.-255.-255.-255")

    # Getting ipv4 dst
    if row["arp_ipv4_dst"] != '':
      rowdict["ipv4_dst"] = row["arp_ipv4_dst"]
    elif row["ethernet_ipv4_dst"] != '':
      rowdict["ipv4_dst"] = row["ethernet_ipv4_dst"]
    elif row['Version'] != '':
      rowdict["ipv4_dst"] = "-1.-1.-1.-1"
    else:
      rowdict["ipv4_dst"] = "-255.-255.-255.-255"
      Warning("No ipv4 dst found and version was '' instead of '6'. setting ipv4 to -255.-255.-255.-255")
    
    if rowdict["ipv4_src"][0:3] == "169":
      rowdict["label"]=1
    else:
      rowdict["label"]=0
    clean_table = clean_table.append(rowdict, ignore_index=True)
    #input()
  
  print(clean_table.columns)
  #input()
  return clean_table, skipped


def max_normalize(dat, maxes=None):
  """
  Normalizes the data by dividing by the max value of
  each column to move the data between 0.0 and 1.0.

  Inputs:
    dat: a numpy array numeric
  Outputs: 
    dat: that array, scaled
    maxes: the numbers that it was scaled by so that
           it can be un-scaled to enterpret the output 
           later. 

  This helps neural networks learn and it makes it so
  that all the features have the same importance initially
  """
  if maxes is None:
    maxes=np.max(dat,axis=0)
  for i in range(maxes.shape[0]):
    if maxes[i]==0:
      maxes[i]=0.001
  dat = dat/maxes
  return dat, maxes

def protocols_to_dummy(prot):
  """
  Turns the protocols I could find into dummy variables.
  This is a one hot encoding

  Inputs: 
    prot: a column of a dataframe that is the 'Protocol'
          attribute from wireshark. Should be a list of
          strings.
  Output:
    dummy_arr: a numpy array with as many columns as there
               are protocols in "prots". if column '1' is
               'ARP' then all of the packets that use the
               'ARP' protocol will have column 1 = 1.0 and
               for those rows, all other protocol columns
               will be 0. This is a one-hot-coding
  """

  prot = pd.DataFrame(prot)
  prots = ["ARP",  "BROWSER",  "DHCP",  "DHCPv6",  "ICMP",  "ICMPv6",  "LLDP",  "LLMNR",  "MDNS",  "MQTT",  "TCP",  "OTHER"]
  dummy_arr = np.zeros((len(prot.index),len(prots)))
  for i in range(len(prot.index)):
    if prot.iat[i,0] in prots:
      dummy_arr[i,indexOf(prots,prot.iat[i,0])] = 1
    else:
      dummy_arr[i,-1]=1
  return dummy_arr


# modified from https://medium.com/mlearning-ai/lstm-with-keras-data-reshaping-in-many-to-one-architecture-c7d3669e3a5c
# No reason to reinvent the wheel on reshaping data
def make_supervised_sequence(data, seq_len, whole_seq_bad=False):
  '''
  Makes data with labels for malicious or not
  input:
    data - the numpy matrix of (n, p+1) shape, where n is the number of rows,
            p+1 is the number of predictors + 1 target column
    k    - the length of the sequence, namely, the number of previous rows 
            (including current) we want to use to predict the target.
    whole_seq_bad 
         - mark any sequence containing a malicious packet as bad instead
           of just the sequences where the newest malicious packet is bad
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

    if whole_seq_bad:
      cur_target = np.max(data[i-seq_len: i, -1])
      
    X_data[i-seq_len,:, :] = cur_sequence.reshape(1, seq_len, X_data.shape[2])
    y_data.append(cur_target)
  
  return X_data, np.asarray(y_data)

def get_modbus_data(cols, file_name, seq_length, train_prop, subset_by_label=-1, maxes=None, verbose=False):
  """
  subset_by_label means that we will get only the data with the desired label
  This is so we can test the unsupervised learner by training it to encode and
  decode normal data and then compare it's ability to decode with that of the 
  anomolous data. This way we can give a model normal real world data with some
  idea of whether it will detect anomolies 
  """
  data_frame = modbus_to_data_frame(file_name)

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
  data, maxes = max_normalize(data, maxes)
  seq_data, seq_labels = make_supervised_sequence(data, seq_length)

  x_train = seq_data[:int(train_prop*seq_data.shape[0]),:,:]
  x_test = seq_data[int(train_prop*seq_data.shape[0]):,:,:]
  y_train = seq_labels[:int(train_prop*seq_labels.shape[0]),]
  y_test = seq_labels[int(train_prop*seq_labels.shape[0]):,]

  return x_train, y_train, x_test, y_test, data_frame, maxes

def get_printer_data(cols, file_name, seq_length, train_prop, subset_by_label=-1, maxes=None, whole_seq_bad=False, verbose=False):
  data_frame, skipped = read_printer_as_df(file_name)

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
  data,maxes = max_normalize(data, maxes)
  #print(data.shape)
  #print(data[0:10,0:10])
  seq_data, seq_labels = make_supervised_sequence(data, seq_length, whole_seq_bad)
  #print(seq_data.shape)
  #print(seq_data[0,0:10,0:10])
  #input("Is this the right shape?")

  x_train = seq_data[:int(train_prop*seq_data.shape[0]),:,:]
  x_test = seq_data[int(train_prop*seq_data.shape[0]):,:,:]
  y_train = seq_labels[:int(train_prop*seq_labels.shape[0]),]
  y_test = seq_labels[int(train_prop*seq_labels.shape[0]):,]

  return x_train, y_train, x_test, y_test, data_frame, maxes
