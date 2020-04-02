import requests
import json
import numpy as np
import pickle
from konduit.client import Client
from konduit.load import client_from_file
import time



client = client_from_file("config.yaml")

root_path = "C:\\Users\\chiaw\\Documents\\data\\konduit-serving-use-cases\\3-2-DL4J-LSTM-PythonClient\\"
featureFile = root_path + "feature.npy"
featureMaskFile = root_path + "featureMask.npy"

print('Load file for class index <> class label')
label_path = "labelclass.pickle"
labelhandler = open(root_path + label_path, 'rb')
labelhandler = pickle.load(labelhandler)

feature = np.load(featureFile, encoding='bytes', allow_pickle=True)
featureMask = np.load(featureMaskFile, encoding='bytes', allow_pickle=True)

prediction = client.predict({client.input_names[0]: feature})

index = int(np.argmax(prediction))

print("Class: {}".format(labelhandler[index]))
print("Probabilities: {}\n".format(np.max(prediction)))
