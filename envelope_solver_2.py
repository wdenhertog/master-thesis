import numpy as np
from numba import njit
import cmath
import matplotlib.pyplot as plt


@njit()
def L_p(k, dr):
    if k == 0:
        return 4 / dr**2
    else:
        return (1 + 1/(2*k)) / dr**2


@njit()
def L_m(k, dr):
    if k == 0:
        return 0.
    else:
        return (1 - 1/(2*k)) / dr**2

@njit()
def L_0(k, dr):
    if k == 0:
        return -4 / dr**2
    else:
        return -2 / dr**2


@njit()
def C_0_p(k, dr, dt, dz, k0p):
    return L_0(k, dr)/2 + 1j * k0p / dt - 3/2 * 1/(dt*dz)# - 1/dt**2


@njit()
def C_0_m(k, dr, dt, dz, k0p):
    return L_0(k, dr)/2 - 1j * k0p / dt + 3/2 * 1/(dt*dz)# - 1/dt**2


@njit()
def theta(n):
    return cmath.phase(n)


@njit()
def D(a_n, j, k, dz):
    return (-3*theta(a_n[j][k]) + 4*theta(a_n[j+1][k]) - theta(a_n[j+2][k])) / (2*dz)


@njit()
def rhs_eq7(a_old, a, a_new, j, k, dt, dr, dz, k0p, nr):
    rhs = (
        #- 2/dt**2 * a[j, k]
        - (C_0_m(k, dr, dt, dz, k0p) - 1j/dt * D(a, j, k, dz)) * a_old[j, k]
        - 2 * np.exp(1j*(theta(a[j, k])-theta(a[j+1, k]))) / (dz*dt) * (a_new[j+1, k] - a_old[j+1, k])
        + np.exp(1j*(theta(a[j, k])-theta(a[j+2, k]))) / (2*dz*dt) * (a_new[j+2, k] - a_old[j+2, k])
        )
    if k+1 < nr:
        rhs -= L_p(k, dr)/2 * a_old[j, k+1]
    if k > 0:
        rhs -= L_m(k, dr)/2 * a_old[j, k-1]
    return rhs


@njit()
def TDMA(a, b, c, d):
    """
    TriDiagonal Matrix Algorithm: solve a linear system Ax=b, where A is a tridiagonal matrix.
    Source: https://stackoverflow.com/questions/8733015/tridiagonal-matrix-algorithm-tdma-aka-thomas-algorithm-using-python-with-nump
    :param a: lower diagonal of A
    :param b: main diagonal of A
    :param c: upper diagonal of A
    :param d: solution vector
    :return: solution to the linear system
    """
    n = len(d)
    w = np.zeros(n - 1, dtype=np.complex128)
    g = np.zeros(n, dtype=np.complex128)
    p = np.zeros(n, dtype=np.complex128)

    w[0] = c[0] / b[0]  # MAKE SURE THAT b[0]!=0
    g[0] = d[0] / b[0]

    for i in range(1, n - 1):
        w[i] = c[i] / (b[i] - a[i - 1] * w[i - 1])
    for i in range(1, n):
        g[i] = (d[i] - a[i - 1] * g[i - 1]) / (b[i] - a[i - 1] * w[i - 1])
    p[n - 1] = g[n - 1]
    for i in range(n - 1, 0, -1):
        p[i - 1] = g[i - 1] - w[i - 1] * p[i]
    return p


@njit()
def solve_2d(anm1, an, nz, nt, nr, zmax, zmin, rmax, dt, k0p):
    dz = (zmax-zmin) / (nz - 1)
    dr = rmax / (nr - 1)

    a_old = np.zeros((nz+2, nr), dtype=np.complex128)
    a = np.zeros((nz+2, nr), dtype=np.complex128)
    a_new = np.zeros((nz+2, nr), dtype=np.complex128)

    a_old[0:-2] = anm1
    a[0:-2] = an

    for n in range(0, nt):
        for j in range(nz-1, -1, -1):  # reversed(range(0, nz)):
            rhs = np.zeros(nr, dtype=np.complex128)
            diag = np.zeros(nr, dtype=np.complex128)
            up_diag = np.zeros(nr-1, dtype=np.complex128)
            lo_diag = np.zeros(nr-1, dtype=np.complex128)
            for k in range(0, nr):
                rhs[k] = rhs_eq7(a_old, a, a_new, j, k, dt, dr, dz, k0p, nr)
                diag[k] = C_0_p(k, dr, dt, dz, k0p) + 1j/dt * D(a, j, k, dz)
                if k < nr-1:
                    up_diag[k] = L_p(k, dr)/2
                if k > 0:
                    lo_diag[k-1] = L_m(k, dr)/2
            a_new[j] = TDMA(lo_diag, diag, up_diag, rhs)
        a_old[:] = a
        a[:] = a_new
    return a_new

