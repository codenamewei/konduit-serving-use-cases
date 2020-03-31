import requests

url = 'http://localhost:65322/raw/json'
obj = {'imagePath': 'C:\\Users\\chiaw\\Desktop\\konduit\\konduit-keras-mnist\\image\\mnist-test1.png'}

response = requests.post(url, json=obj)

if response.status_code == 200:

    content = response.json()
    output_class = content['output_class']
    output = content['output']

    print(output_class)
    print(output)

else:
    print(response)
