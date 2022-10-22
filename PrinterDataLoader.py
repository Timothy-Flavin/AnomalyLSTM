from asyncore import read
from distutils.command.clean import clean
from operator import indexOf
import pandas as pd
import numpy as np
from DataLoader import eth_to_numpy, ip_to_numpy

# Version is either 0 for ipv4 or 1 for ipv6
def read_printer_as_df(path="Data/PrinterPackets.csv"):
  """
  Reads the printer data csv 
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


def max_normalize(dat):
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
