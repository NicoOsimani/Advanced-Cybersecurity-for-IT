__author__ = 'Daniele Marzetti'
from preprocess import *
from model import *

dataset, label, MAX_STRING_LENGTH = create_set_label()
dataset_preprocessed = to_numeric(dataset, MAX_STRING_LENGTH)