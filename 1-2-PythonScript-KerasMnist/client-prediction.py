import requests

url = 'http://localhost:65322/raw/json'
obj = {'imagePath': 'C:\\Users\\chiaw\\Documents\\data\\konduit-serving-use-cases\\2-1-Keras-CNN-PythonClient\\image\\mnist-test1.png'}

response = requests.post(url, json=obj)

if response.status_code == 200:

    content = response.json()
    output_class = content['output_class']
    output = content['output']

    print("output {}".format(output))
    print("output_class {}".format(output_class))

else:
    print(response)
