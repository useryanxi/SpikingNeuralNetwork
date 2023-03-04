from snn.neurons.LeakyIntegrateAndFireNeuron import LeakyIntegrateAndFireNeuron
from snn.learning.stdp import STDP
from snn.network.snn import SNN
import numpy as np
import imageio.v2 as imageio
import os
from rf import rf
from encode import encode
import pickle as pk


net = SNN(784, [], 10, LeakyIntegrateAndFireNeuron, STDP())

path = r"./mnist_png/training"
counter = int(input("How many loops : "))
img_num = int(input("How many img : "))
T = int(input("Time scale : "))

for _ in range(counter):
    for folder in os.listdir(path):
        files = os.listdir(os.path.join(path, folder))[:img_num]
        n_files = len(files)
        print(f"- Start : '{folder}' with {n_files} files")
        for i, img in enumerate(files):
            img_path = os.path.join(os.path.join(path, folder), img)
            train = np.array(encode(rf(imageio.imread(img_path)), T=T))
            for t in range(len(train[0])):
                net.solve(list(train[:, t]))
            if i>0 and i % (n_files / 10) == 0:
                print(f"Done at {i * 100 / n_files:.2f}%")

with open("snn.pickle", "wb") as file:
    pk.dump(net, file)
    print("Export Done")
