# Copyright (c) 2020, codenamewei
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
from PIL import Image
from konduit.load import client_from_file

im = Image.open('5_32x32.png').convert('L')  # to gray image .convert('LA')

# channels_first -> [batch_size, 1, img_rows, img_cols]
im = np.array(im.resize((28, 28)))
im = np.expand_dims(im, axis=0)
im = np.expand_dims(im, axis=1)
# print(im.shape)

client = client_from_file("config.yaml")

files = {client.input_names[0]: im}

prediction = client.predict(files)

print("Class: {}".format(np.argmax(prediction, 1)[0]))
print("Probabilities {:0.1f}".format(np.max(prediction, 1)[0]))

print('Program end...')
