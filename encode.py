import numpy as np
from numpy import interp
from math import ceil


def encode(pot, pixel_x=28, T=350):
    # initializing spike train
    train = []
    for l in range(pixel_x):
        for m in range(pixel_x):
            temp = np.zeros([(T + 1)])
            # calculating firing rate proportional to the membrane potential
            freq = interp(pot[l][m], [np.min(pot), np.max(pot)], [1, 50])
            time_period = ceil(T / freq)
            # generating spikes according to the firing rate
            time_of_spike = time_period
            if pot[l][m] > 0:
                while time_of_spike < (T + 1):
                    temp[int(time_of_spike)] = 1
                    time_of_spike += time_period
            train.append(temp)
    return train

# # Not finished
# def encode_bin(pot, pixel_x=28, T=350):
#     train = []
#     for i in pot:
#         for j in i:
#             val = [int(i) for i in bin(j).replace("0b", "")]
#             extra_zeros = (8 - len(val)) * [0] 
#             val = extra_zeros + val
#             train += val
#     return train

# def decode_bin(pot):
#     total = 0
#     pos = len(pot) - 1
#     for i in pot:
#         total += i * (2 ** pos)
#         pos -= 1
#     return total
