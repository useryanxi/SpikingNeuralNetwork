from snn.neurons.LeakyIntegrateAndFireNeuron import LeakyIntegrateAndFireNeuron
from snn.learning.stdp import STDP
from snn.network.snn import SNN
import numpy as np
import imageio.v2 as imageio
import os
from rf import rf
from encode import encode


net = SNN(3, [2, 4, 5], 6, LeakyIntegrateAndFireNeuron, STDP())

counter = int(input("How many loops : "))
# img_num = int(input("How many img : "))
path =r"./mnist_png/training"
for _ in range(counter):
    for folder in os.listdir(path):
        for img in os.listdir(os.path.join(path, folder)):
            img_path = os.path.join(os.path.join(path, folder), img)
            train = np.array(encode(rf(imageio.imread(img_path))))
            for i in train:
                print(i)
            quit()

# Weights
print("Weights:")
for i, layer in enumerate(net.layers):
    if i == 0:
        print("Input", layer[0].weights)
    elif layer is net.layers[-1]:
        print("Output", layer[0].weights)
    else:
        print(f"Hidden layer {i}", layer[0].weights)
