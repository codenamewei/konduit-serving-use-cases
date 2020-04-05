import requests
import json
import numpy as np
import pickle
from konduit.client import Client
from konduit.load import client_from_file
import time

def get_output(response, response_time):

    index = int(np.argmax(response))

    print("Class: {}".format(labelhandler[index]))
    print("Probabilities: {}\n".format(np.max(prediction)))


    print("Average time per request: {0:.2f} ms/request".format(response_time))
    print("Average time per request: {0:.2f} s/request".format(response_time * 0.001))

current_milli_time = lambda: int(round(time.time() * 1000))

port = 65322

client = client_from_file("config.yaml")

root_path = "C:\\Users\\chiaw\\Documents\\data\\"

print('Load file for class index <> class label')
labelhandler = open(root_path + "konduit-serving-use-cases\\3-3-DL4J-LSTMTextClassification-MultiSteps\\labelclass.pickle", 'rb')
labelhandler = pickle.load(labelhandler)

f = open (root_path + "20news-bydate\\20news-bydate-test\\alt.atheism\\53068", "r")

contents = f.read()


payload = {client.input_names[0]: contents}

#-----Konduit Package ------------

before_milli_time = current_milli_time()

prediction = client.predict(payload)

after_milli_time = current_milli_time()

get_output(prediction, after_milli_time - before_milli_time)

#-----Requests Package ------------
before_milli_time = current_milli_time()

response = requests.post("http://localhost:65322/raw/json", json = payload)

after_milli_time = current_milli_time()

if response.status_code == 200:
    print('\nRaw Requests Success!\n')
    get_output(response, after_milli_time - before_milli_time)
elif response.status_code == 404:
    print('RAW Request failed.')
