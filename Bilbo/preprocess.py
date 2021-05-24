__author__ = 'Daniele Marzetti'
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def create_set_label():
  dataset = pd.read_csv("/content/drive/MyDrive/Cyber Security/dga_domains_full.csv", encoding= "utf-8", names=['label', 'family', 'domain'])
  dataset.drop(columns='family', inplace=True)
  label = pd.Series(dataset['label'] == 'dga', dtype=int)
  dataset.drop(columns='label', inplace=True)
  dataset = pd.Series(dataset['domain'])
  dataset.to_csv("/content/drive/MyDrive/Cyber Security/dataset.csv", header=False, index=False)
  label.to_csv("/content/drive/MyDrive/Cyber Security/label.csv", header=False, index=False)
  return dataset, label, dataset.map(len).max()


def conversion(x, mapping):
  converted = []
  for y in list(x):
    converted.append(mapping.get(y))
  return converted

def to_numeric(dataset, MAX_STRING_LENGTH):
  valid_characters = set()
  dataset_characters = dataset.map(list)
  [valid_characters.add(j) for i in dataset_characters for j in i]

  charachetrs_map = dict.fromkeys(valid_characters)

  for i in charachetrs_map.keys():  #ordinamento personalizzato
    if ord(i)>47 and ord(i)<58:
      charachetrs_map[i]=ord(i)-21
    if ord(i)>96:
      charachetrs_map[i]=ord(i)-96

  MAX_INDEX = max((filter(None.__ne__,list(charachetrs_map.values()))))

  for i in sorted(charachetrs_map.keys()):
    if charachetrs_map[i]==None:
      MAX_INDEX+=1
      charachetrs_map[i]=MAX_INDEX

  dataset_preprocess = dataset.apply(conversion, mapping = charachetrs_map)
  dataset_final = pad_sequences(dataset_preprocess.to_numpy(), maxlen=MAX_STRING_LENGTH, padding="pre", value=0)
  return  dataset_final, MAX_INDEX + 1