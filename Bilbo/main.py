__author__ = 'Daniele Marzetti'
from preprocess import *
from network import *

dataset, label, MAX_STRING_LENGTH = create_set_label()
dataset_preprocessed, MAX_INDEX = to_numeric(dataset, MAX_STRING_LENGTH)
model = create_model(MAX_STRING_LENGTH, MAX_INDEX)
train_eval_test(model, dataset, label)