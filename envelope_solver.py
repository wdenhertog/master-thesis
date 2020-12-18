import cmath
import numpy as np


def Lk(sign, k, dp):
    """
    Calculation of L_k^{+-}
    :param sign: value -1, 0 or 1, which defines the -, 0, + symbol respectively
    :param k: the rho grid coordinate: 0<=k<=Np-1
    :param dp: grid size in rho direction
    :return: L_k^{+-}
    """
    if k > 0:
        if sign == 1 or sign == -1:
            return (1 + sign * 1 / (2 * k)) / dp ** 2
        else:
            return -2 / dp ** 2
    else:
        if sign == 1:
            return 4 / dp ** 2
        elif sign == 0:
            return -4 / dp ** 2
        else:
            return 0


def Ck(sign, k, k0p, dt, dz, dp):
    """
    Calculate Equation (8) from Benedetti - 2018
    :param sign: 1 or -1, which corresponds to + or - respectively in Equation (8)
    :param k: the rho grid coordinate: 0<=k<=Np-1
    :param k0p: k0/kp
    :param dt: tau step
    :param dz: zeta step
    :param dp: rho step
    :return: C_k^{0,+-}
    """
    # TODO: add 2nd time derivative
    return Lk(-1, k, dp) / 2 + sign * 1j * k0p * 1 / dt - sign * 3 / 2 * 1 / (dt * dz)


def theta(a, j):
    """
    Calculate the phase of â
    :param a: vector â at a specific time t
    :param j: the zeta grid coordinate: 0<=j<nz
    :return: phase of â in â(z, t)
    """
    return cmath.phase(a[j])


def D(a, j, dz):
    """
    Calculate D in Equation (6) from Benedetti - 2018
    :param a: vector â at a specific time t
    :param j: the zeta grid coordinate: 0<=j<nz
    :param dz: zeta step
    :return: D_{j}^n
    """
    return (-3 * theta(a, j) + 4 * theta(a, j + 1) - theta(a, j + 2)) / (2 * dz)


def chi(j, k, n):
    return 0  # TODO: implement calculation of chi at time n and position (z,r) = (j,k).


def solve_1d(k0p, zmin, zmax, nz, dt, nt, a0):
    """
    Solve the 1D envelope equation, without second time derivative and without chi:
    (2i*k0/kp*d/dt+2*d^2/(dzdt))â = 0
    :param k0p: k0/kp = central laser wavenumber (2pi/lambda_0) / plasma skin depth
    :param zmin: minimum value for zeta (nondimensionalized z-direction, zeta=k_p(z-ct))
    :param zmax: maximum value for zeta
    :param nz: number of grid points in zeta-direction
    :param dt: size of tau step (nondimensionalized time, tau=k_pct)
    :param nt: number of tau steps
    :param a0: value for â(z,0), array of dimensions (nz, 1)
    :return: a[z][t], 2D array of the value of â at every point zmin<=z<=zmax, and 0<=t<=dt*nt
    """
    # Reserve 2 rows for the ghost grid points at j = nz and j = nz+1.
    # a_old corresponds to a(z, t-1), a_current corresponds to a(z, t) and a_new corresponds to a(z, t+1)
    # The change from coordinates in a_current is as follows: a[j] corresponds to a(zmin + j*dz, n).
    a_old = np.concatenate((a0, np.zeros(2))).astype(complex)  # TODO: change a(z,-1) using Gaussian wave equation
    a_current = np.concatenate((a0, np.zeros(2))).astype(complex)
    a_new = np.zeros(nz + 2, dtype=complex)

    dz = (zmax - zmin) / (nz - 1)

    c0p = Ck(1, 0, k0p, dt, dz, 0)  # C_0^{0, +}
    c0m = Ck(-1, 0, k0p, dt, dz, 0)  # C_0^{0, -}

    for n in range(0, nt):
        for j in range(nz, 0, -1):
            factor_lhs = c0p + 1j / dt * D(a_current, j - 1, dz)
            factor_rhs = -(c0m - 1j / dt * D(a_current, j - 1, dz)) \
                         * a_old[j - 1] \
                         - 2 * np.exp(1j * (theta(a_current, j - 1) - theta(a_current, j))) / (dt * dz) \
                         * (a_new[j] - a_old[j]) \
                         + np.exp(1j * (theta(a_current, j - 1) - theta(a_current, j + 1))) / (dt * dz) \
                         * (a_new[j + 1] - a_old[j + 1])
            a_new[j - 1] = factor_rhs / factor_lhs
        a_old = a_current
        a_current = a_new
    return a_new
