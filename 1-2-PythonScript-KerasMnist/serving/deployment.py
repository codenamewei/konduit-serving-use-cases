import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img

root_path = "C:\\Users\\chiaw\\Documents\\data\\konduit-serving-use-cases\\2-1-Keras-CNN-PythonClient\\"
model = load_model(root_path + 'keras_cnn_model.h5')

#imagePath = root_path + 'image\\mnist-test1.png'
img = load_img(imagePath, False,'grayscale',target_size=(28,28))

x = img_to_array(img)
x = x.astype('float32')
x /= 255
x = np.expand_dims(x, axis=0)

np_output = model.predict(x)# output of type numpy ndarray

output = str(np_output.tolist()[0])

prediction_class = model.predict_classes(x)
output_class = str(prediction_class.item(0))

print("output {}".format(output))
print("output_class {}".format(output_class))

K.clear_session()

print("End of program...")
