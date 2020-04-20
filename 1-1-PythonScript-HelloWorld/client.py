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

import json
import time
import numpy as np
from konduit.client import Client
from konduit.load import client_from_file

current_milli_time = lambda: int(round(time.time() * 1000))

client = client_from_file("config.yaml")

prediction = client.predict({client.input_names[0]: np.array([5])})

print("Output: {}".format(prediction))
