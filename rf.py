import numpy as np


def rf(inp, pixel_x=28):
    sca1 = 0.625
    sca2 = 0.125
    sca3 = -0.125
    sca4 = -0.5

    # Receptive field kernel
    w = [
        [sca4, sca3, sca2, sca3, sca4],
        [sca3, sca2, sca1, sca2, sca3],
        [sca2, sca1, 1.00, sca1, sca2],
        [sca3, sca2, sca1, sca2, sca3],
        [sca4, sca3, sca2, sca3, sca4],
    ]

    pot = np.zeros([pixel_x, pixel_x])
    ran = [-2, -1, 0, 1, 2]
    ox = 2
    oy = 2

    # Convolution
    for i in range(pixel_x):
        for j in range(pixel_x):
            summ = 0
            for m in ran:
                for n in ran:
                    if (
                        (i + m) >= 0
                        and (i + m) <= pixel_x - 1
                        and (j + n) >= 0
                        and (j + n) <= pixel_x - 1
                    ):
                        summ = summ + w[ox + m][oy + n] * inp[i + m][j + n] / 255
            pot[i][j] = summ
    return pot
