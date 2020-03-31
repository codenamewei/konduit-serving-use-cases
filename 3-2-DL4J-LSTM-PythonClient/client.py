import requests
import json
import numpy as np
import pickle
from konduit.client import Client
from konduit.load import client_from_file
import time

client = client_from_file("config.yaml")

root_path = "C:\\Users\\chiaw\\Documents\\data\\konduit-serving-use-cases\\3-2-DL4J-LSTM-PythonClient\\"
featureFile = root_path + "alt-atheism-feature.npy"
featureMaskFile = root_path + "alt-atheism-featureMask.npy"

feature = np.load(featureFile, encoding='bytes', allow_pickle=True)
featureMask = np.load(featureMaskFile, encoding='bytes', allow_pickle=True)

prediction = client.predict({client.input_names[0]: feature})

print(prediction)
