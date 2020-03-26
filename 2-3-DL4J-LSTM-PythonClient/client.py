import requests
import json
import numpy as np
import pickle
from konduit.client import Client
from konduit.load import client_from_file
import time

current_milli_time = lambda: int(round(time.time() * 1000))

port = 65322

client = client_from_file("config.yaml")

print('Load file for class index <> class label')
label_path = "labelclass.pickle"
labelhandler = open(label_path, 'rb')
labelhandler = pickle.load(labelhandler)

root_path = "D:\\Users\\chiawei\\Documents\\data\\newsgroup_data\\20news-bydate\\"
f = open (root_path + "20news-bydate-test\\alt.atheism\\53280", "r")

contents = f.read()

input_param = {"String": contents}

total_count = 5
iteration = range(total_count)

before_milli_time = current_milli_time()

prediction = client.predict({client.input_names[0]: input_param})

print(prediction)

output = np.array(prediction[client.output_names[0]]["ndArray"]["data"])

index = int(np.argmax(output, 0))

print("Class: {}".format(labelhandler[index]))
print("Probabilities: {}\n".format(np.max(output[index])))

after_milli_time = current_milli_time()

total_time = after_milli_time - before_milli_time
milli_average = total_time / float(total_count)

print("Total time taken: {} ms".format(total_time))
print("Average time per request: {0:.2f} ms/request".format(milli_average))
print("Average time per request: {0:.2f} s/request".format(milli_average * 0.001))
