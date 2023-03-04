import os
import numpy as np
import pickle as pk
import imageio.v2 as imageio
from rf import rf
from encode import encode

with open("snn.pickle", "rb") as file:
    net = pk.load(file)
    print("Import Done")

path = r"./mnist_png/testing"
# img_num = int(input("How many img : "))
# T = int(input("Time scale : "))

print(net.layers[-1])
print(len(net.layers[-1]))
print(net.layers[-1][0],net.layers[-1][0].weights)
print(net.layers[0][1],net.layers[-1][1].weights)


for folder in os.listdir(path):
    files = os.listdir(os.path.join(path, folder))[:1]
    n_files = len(files)
    print(f"- Start : '{folder}' with {n_files} files")
    for i, img in enumerate(files):
        img_path = os.path.join(os.path.join(path, folder), img)
        train = np.array(encode(rf(imageio.imread(img_path)), T=20))
        info = []
        for t in range(len(train[0])):
            info = net.solve(list(train[:, t]))
            print(info)
        print(info)
