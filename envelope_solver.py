import cmath
import numpy as np


def Lk(sign, k, dp):
    """
    Calculation of L_k^{+-}
    :param sign: value -1, 0 or 1, which defines the -, 0, + symbol respectively
    :param k: the rho grid coordinate: 0<=k<=Np-1
    :param dp: grid size in rho direction
    :return: output defined as in Benedetti - 2018
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


def theta(a, j, n):
    return cmath.phase(a[j][n])


def D(a, j, n, dz):
    return (-3 * theta(a, j, n) + 4 * theta(a, j + 1, n) - theta(a, j + 2, n)) / (2 * dz)


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
    # reserve 2 rows for the ghost grid points at j = nz and j = nz+1 and 2 columns for t = -nt and t = (nt+1)*dt
    # so now the first column of a is 0, and the second column of a is the initial condition.
    a = np.zeros((nz + 2, nt + 2), dtype=complex)
    a[:, 0] = a[:, 1] = np.concatenate((a0, np.zeros(2)))

    dz = (zmax - zmin) / (nz - 1)

    c0p = Ck(1, 0, k0p, dt, dz, 0)  # C_0^{0, +}
    c0m = Ck(-1, 0, k0p, dt, dz, 0)  # C_0^{0, -}

    for n in range(1, nt):
        for j in range(nz, 0, -1):
            factor_lhs = c0p + 1j / dt * D(a, j - 1, n, dz)
            factor_rhs = -(c0m - 1j / dt * D(a, j - 1, n, dz)) * a[j - 1][n - 1] - 2 * np.exp(
                1j * (theta(a, j - 1, n) - theta(a, j, n))) / (dt * dz) * (a[j][n + 1] - a[j][n - 1]) + np.exp(
                1j * (theta(a, j - 1, n) - theta(a, j + 1, n))) / (dt * dz) * (a[j + 1][n + 1] - a[j + 1][n - 1])
            a[j - 1][n + 1] = factor_rhs / factor_lhs
    return a
