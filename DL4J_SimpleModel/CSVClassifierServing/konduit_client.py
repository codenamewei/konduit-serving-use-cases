import json
import numpy as np
from konduit.client import Client
from konduit.load import client_from_file

current_milli_time = lambda: int(round(time.time() * 1000))

client = client_from_file("graph_config.yaml")

total_count = 5
iteration = range(total_count)

before_milli_time = current_milli_time()

arr_input = [0.272702493273322,0.0201936700818061] #label: 0
#arr_input = [0.867855051798607,0.597829895646543] #label: 1

for i in iteration:

    prediction = client.predict({client.input_names[0]: arr_input})

    print(prediction)

    #output = np.array(prediction[client.output_names[0]]["ndArray"]["data"])

    #index = int(np.argmax(output, 0))

after_milli_time = current_milli_time()

total_time = after_milli_time - before_milli_time
milli_average = total_time / float(total_count)

print("Total time taken: {} ms".format(total_time))
print("Average time per request: {0:.2f} ms/request".format(milli_average))
print("Average time per request: {0:.2f} s/request".format(milli_average * 0.001))
