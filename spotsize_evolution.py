import numpy as np
import matplotlib.pyplot as plt


def spotsize_rel(dn_c, dn, r_0, w_0, k_os, z):
    # returns 2*w^2/w_0^2
    return 1 + dn_c * r_0 ** 4 / (dn * w_0 ** 4) + (
            1 - dn_c * r_0 ** 4 / (dn * w_0 ** 4)) * np.cos(k_os * z)


def spotsize(dn_c, dn, r_0, w_0, k_os, z):
    # returns w
    return np.sqrt((1 + dn_c * r_0 ** 4 / (dn * w_0 ** 4) + (
                1 - dn_c * r_0 ** 4 / (dn * w_0 ** 4)) * np.cos(
        k_os * z)) * w_0 ** 2 / 2)


def pulse_width(a):
    # return position where the pulse is lower than max / e.
    array_sum = np.sum(np.abs(a), axis=0)
    pulse_max = max(array_sum)
    threshold = pulse_max / np.e
    return next(x[0] for x in enumerate(array_sum) if x[1] < threshold)


if __name__ == '__main__':
    r_0 = 19e-6
    w_0 = 20e-6
    dn_c = 1 / (np.pi * 2.8179403262e-15 * r_0 ** 2)
    dn = dn_c
    l_0 = 0.8e-6
    z_m = np.pi * r_0 ** 2 / l_0
    k_os = (2 / z_m) * np.sqrt(dn / dn_c)
    z = np.linspace(0, 0.01, 1000)

    w_rel = spotsize_rel(dn_c, dn, r_0, w_0, k_os, z)
    w = spotsize(dn_c, dn, r_0, w_0, k_os, z)
   # plt.plot(z, w_rel)
    plt.plot(z, w)
