import numpy as np


def spotsize(dn_c, dn, r_0, w_0, k_os, z):
    return 1 + dn_c * r_0 ** 4 / (dn * w_0 ** 4) + (
            1 - dn_c * r_0 ** 4 / (dn * w_0 ** 4)) * np.cos(k_os * z)


if __name__ == '__main__':
    r_0 = 1
    dn_c = 1 / (np.pi * 2.8179403262e-15 * r_0 ** 2)
    dn = 1
    w_0 = 1
    l_0 = 1
    z_m = np.pi * r_0 ** 2 / l_0
    k_os = (2/z_m)*np.sqrt(dn/dn_c)
    z = np.linspace(0, 10, 100)

    w = spotsize(dn_c, dn, r_0, w_0, k_os, z)
