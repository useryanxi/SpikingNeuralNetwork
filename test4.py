from snn.neurons.LeakyIntegrateAndFireNeuron import LeakyIntegrateAndFireNeuron
from snn.learning.stdp import STDP
from snn.network.snn import SNN
import numpy as np
import imageio.v2 as imageio
import os
from rf import rf
from encode import encode, encode_bin
import time

img_time = True
read_time = True

t_begin = time.time()
net = SNN(784, [28, 28, 10, 10], 10, LeakyIntegrateAndFireNeuron, STDP())

counter = int(input("How many loops : "))
img_num = int(input("How many img : "))
T = 28 * 28 * 8

# Train
t_train_b = time.time()
path = r"./mnist_png/training"
for _ in range(counter):
    for folder in os.listdir(path):
        files = os.listdir(os.path.join(path, folder))[:img_num]
        n_files = len(files)
        print(f"- Start : '{folder}' with {n_files} files")
        for i, img in enumerate(files):
            img_path = os.path.join(os.path.join(path, folder), img)
            if img_time:
                t_conv = time.time()
            train = np.array(encode_bin(imageio.imread(img_path), T=T))
            if img_time:
                img_time = False
                print(f"Conversion to spike in {time.time() - t_conv}s")
            print(len(train), len(train[0]))
            if read_time:
                t_read = time.time()
            for t in range(len(train[0])):
                net.solve(list(train[:, t]))
            if read_time:
                read_time = False
                print(f"Reading Spikes in {time.time() - t_read}s")
            if i % (n_files / 10) == int(n_files / 10) - 1:
                print(f"Done at {i * 100 / n_files:.2f}%")
print(f"Trained in {time.time() - t_train_b}s")

# Test
read_time = True
t_test_b = time.time()
path = r"./mnist_png/testing"
for folder in os.listdir(path):
    files = os.listdir(os.path.join(path, folder))[:img_num]
    n_files = len(files)
    print(f"- Start : '{folder}' with {n_files} files")
    for i, img in enumerate(files):
        img_path = os.path.join(os.path.join(path, folder), img)
        train = np.array(encode_bin(rf(imageio.imread(img_path)), T=T))
        if read_time:
            t_read = time.time()
        for t in range(len(train[0])):
            print(net.test(list(train[:, t])))
        if read_time:
            read_time = False
            print(f"Load tested image in {time.time() - t_read}s")
print(f"Tested in {time.time() - t_test_b}s")
print(f"Done in {time.time() - t_begin}s")
