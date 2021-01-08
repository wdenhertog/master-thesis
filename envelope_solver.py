import cmath
import numpy as np
from numba import njit, prange


@njit()
def testfunc(z, zm):
    return -4 * (z - zm)**2


@njit()
def L(sign, k, dr):
    """
    Calculation of L_k^{+-}
    :param sign: value -1, 0 or 1, which defines the -, 0, + symbol respectively
    :param k: the rho grid coordinate: 0<=k<=Np-1
    :param dr: grid size in rho direction
    :return: L_k^{+-}
    """
    if k > 0:
        if sign == 1 or sign == -1:
            return (1 + sign * 1 / (2 * k)) / dr ** 2
        else:
            return -2 / dr ** 2
    else:
        if sign == 1:
            return 4 / dr ** 2
        elif sign == 0:
            return -4 / dr ** 2
        else:
            return 0


@njit()
def C(sign, k, k0p, dt, dz, dr):
    """
    Calculate Equation (8) from Benedetti - 2018
    :param sign: 1 or -1, which corresponds to + or - respectively in Equation (8)
    :param k: the rho grid coordinate: 0<=k<=Np-1
    :param k0p: k0/kp
    :param dt: tau step
    :param dz: zeta step
    :param dr: rho step
    :return: C_k^{0,+-}
    """
    if dr == 0:
        return L(-1, k, dr) / 2 + sign * 1j * k0p * 1 / dt - sign * 3 / 2 * 1 / (dt * dz)  # 1D case
    else:
        return L(0, k, dr) / 2 + sign * 1j * k0p * 1 / dt - sign * 3 / 2 * 1 / (dt * dz) - 1 / dt ** 2  # 2D case


@njit()
def theta1D(a, j):
    """
    Calculate the phase of â in the 1D model
    :param a: vector â at a specific time t
    :param j: the zeta grid coordinate: 0<=j<nz
    :return: phase of â in â(z, t)
    """
    return cmath.phase(a[j])


@njit()
def theta(a, j, k):
    """
    Calculate the phase of â in the 2D model
    :param a: matrix â at a specific time t
    :param j: the zeta grid coordinate: 0<=j<nz
    :param k: the rho grid coordinate: 0<=k<nr
    :return: phase of â in â(z, r, t)
    """
    return cmath.phase(a[j][k])


@njit()
def D1D(a, j, dz):
    """
    Calculate D in Equation (6) from Benedetti - 2018 (1D case)
    :param a: vector â at a specific time t
    :param j: the zeta grid coordinate: 0<=j<nz
    :param dz: zeta step
    :return: D_{j}^n
    """
    return (-3 * theta1D(a, j) + 4 * theta1D(a, j + 1) - theta1D(a, j + 2)) / (2 * dz)


@njit()
def D(a, j, k, dz):
    """
    Calculate D in Equation (6) from Benedetti - 2018
    :param a: matrix â at a specific time t
    :param j: the zeta grid coordinate: 0<=j<nz
    :param k: the rho grid coordinate: 0<=k<nr
    :param dz: zeta step
    :return: D_{j}^n
    """
    return (-3 * theta(a, j, k) + 4 * theta(a, j + 1, k) - theta(a, j + 2, k)) / (2 * dz)


@njit()
def chi(j, k, n):
    return 0  # TODO: implement calculation of chi at time n and position (z,r) = (j,k).


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
    :return: a[z], 1D array of the value of â at every point zmin<=z<=zmax, and t = dt*nt
    """
    # Reserve 2 rows for the ghost grid points at j = nz and j = nz+1.
    # a_old corresponds to a(z, t-1), a_current corresponds to a(z, t) and a_new corresponds to a(z, t+1)
    a_old = np.concatenate((a0, np.zeros(2))).astype(complex)
    a_current = np.concatenate((a0, np.zeros(2))).astype(complex)
    a_new = np.zeros(nz + 2, dtype=complex)

    dz = (zmax - zmin) / (nz - 1)

    c0p = C(1, 0, k0p, dt, dz, 0)  # C_0^{0, +}
    c0m = C(-1, 0, k0p, dt, dz, 0)  # C_0^{0, -}

    for n in range(0, nt):
        for j in range(nz, 0, -1):
            factor_lhs = c0p + 1j / dt * D1D(a_current, j - 1, dz)
            factor_rhs = (-(c0m - 1j / dt * D1D(a_current, j - 1, dz))
                          * a_old[j - 1]
                          - 2 * np.exp(1j * (theta1D(a_current, j - 1) - theta1D(a_current, j))) / (dt * dz)
                          * (a_new[j] - a_old[j])
                          + np.exp(1j * (theta1D(a_current, j - 1) - theta1D(a_current, j + 1))) / (2 * dt * dz)
                          * (a_new[j + 1] - a_old[j + 1]))
            a_new[j - 1] = factor_rhs / factor_lhs
        a_old = a_current
        a_current = a_new
    return a_new


@njit(parallel=True)
def solve_2d(k0p, zmin, zmax, nz, dt, nt, rmax, nr, a0, aold):
    """
    Solve the 2D envelope equation (\nabla_tr^2+2i*k0/kp*d/dt+2*d^2/(dzdt)-d^2/dt^2)â = \chi*â
    :param k0p: k0/kp = central laser wavenumber (2pi/lambda_0) / plasma skin depth
    :param zmin: minimum value for zeta (nondimensionalized z-direction, zeta=k_p(z-ct))
    :param zmax: maximum value for zeta
    :param nz: number of grid points in zeta-direction
    :param dt: size of tau step (nondimensionalized time, tau=k_pct)
    :param nt: number of tau steps
    :param rmax: maximum value for rho (minimum value is always 0). rho = k_p*r (nondimensionalized radius)
    :param nr: number of rho steps
    :param a0: value for â(z,r,0), array of dimensions (nz, nr)
    :param aold: value for â(z,r,-1), array of dimensions (nz, nr)
    :return: a[z][r], 2D array of the value of â at every point zmin<=z<=zmax, 0<=r<rmax and t = dt*nt
    """
    # a_old corresponds to a(z, r, t-1), a_current corresponds to a(z, r, t) and a_new corresponds to a(z, r, t+1)
    a_old = np.zeros((nz + 2, nr), dtype=np.complex128)  # add 2 rows of ghost points in the zeta direction
    a_current = np.zeros((nz + 2, nr), dtype=np.complex128)
    a_new = np.zeros((nz + 2, nr), dtype=np.complex128)
    a_old[0:-2] = aold
    a_current[0:-2] = a0

    dz = (zmax - zmin) / (nz - 1)
    dr = rmax / (nr - 1)

    for t in range(0, nt):
        if t % 100 == 0:
            print("Time =", t * dt)
        # For every j, solve the tridiagonal system to calculate the solution over on the radius
        for j in range(nz, 0, -1):
            d_upper = np.zeros(nr - 1, dtype=np.complex128)
            d_lower = np.zeros(nr - 1, dtype=np.complex128)
            d_main = np.zeros(nr, dtype=np.complex128)
            sol = np.zeros(nr, dtype=np.complex128)
            # boundary conditions for k = 0 and k = nr - 1
            d_main[0] = C(1, 0, k0p, dt, dz, dr) - chi(j - 1, 0, t) / 2 + 1j / dt * D(a_current, j - 1, 0, dz)
            d_main[-1] = (C(1, nr - 1, k0p, dt, dz, dr)
                          - chi(j - 1, nr - 1, t) / 2 + 1j / dt * D(a_current, j - 1, nr - 1, dz))
            sol[0] = (-2 / dt ** 2 * a_current[j - 1][0]
                      - (C(-1, 0, k0p, dt, dz, dr) - chi(j - 1, 0, t) - 1j / dt * D(a_current, j - 1, 0, dz))
                      * a_old[j - 1][0]
                      - L(1, 0, dr) / 2 * a_old[j - 1][1]
                      - 2 * np.exp(1j * (theta(a_current, j - 1, 0) - theta(a_current, j, 0))) / (dt * dz)
                      * (a_new[j][0] - a_old[j][0])
                      + np.exp(1j * (theta(a_current, j - 1, 0) - theta(a_current, j + 1, 0))) / (2 * dt * dz)
                      * (a_new[j + 1][0] - a_old[j + 1][0]))
            sol[-1] = (-2 / dt ** 2 * a_current[j - 1][-1]
                       - L(-1, nr - 1, dr) / 2 * a_old[j - 1][-2]
                       - (C(-1, nr - 1, k0p, dt, dz, dr) - chi(j - 1, nr - 1, t) - 1j / dt
                          * D(a_current, j - 1, nr - 1, dz))
                       * a_old[j - 1][-1]
                       - 2 * np.exp(1j * (theta(a_current, j - 1, nr - 1) - theta(a_current, j, nr - 1))) / (dt * dz)
                       * (a_new[j][-1] - a_old[j][-1])
                       + np.exp(1j * (theta(a_current, j - 1, nr - 1) - theta(a_current, j + 1, nr - 1)))
                       / (2 * dt * dz)
                       * (a_new[j + 1][-1] - a_old[j + 1][-1]))
            for k in prange(1, nr - 1):
                d_main[k] = C(1, k, k0p, dt, dz, dr) - chi(j - 1, k, t) / 2 + 1j / dt * D(a_current, j - 1, k, dz)
                sol[k] = (-2 / dt ** 2 * a_current[j - 1][k]
                          - L(-1, k, dr) / 2 * a_old[j - 1][k - 1]
                          - (C(-1, k, k0p, dt, dz, dr) - chi(j - 1, k, t) - 1j / dt * D(a_current, j - 1, k, dz))
                          * a_old[j - 1][k]
                          - L(1, k, dr) / 2 * a_old[j - 1][k + 1]
                          - 2 * np.exp(1j * (theta(a_current, j - 1, k) - theta(a_current, j, k))) / (dt * dz)
                          * (a_new[j][k] - a_old[j][k])
                          + np.exp(1j * (theta(a_current, j - 1, k) - theta(a_current, j + 1, k))) / (2 * dt * dz)
                          * (a_new[j + 1][k] - a_old[j + 1][k]))
                d_lower[k - 1] = L(-1, k, dr) / 2
                d_upper[k - 1] = L(1, k - 1, dr) / 2
            d_lower[-1] = L(-1, nr - 1, dr) / 2
            d_upper[-1] = L(1, nr - 2, dr) / 2
            a_new[j - 1] = TDMA(d_lower, d_main, d_upper, sol)
        a_old = a_current
        a_current = a_new
    return a_new


@njit(parallel=True)
def solve_2d_test(k0p, zmin, zmax, nz, dt, nt, rmax, nr, a0, aold):
    """
    Solve the 2D envelope equation (\nabla_tr^2+2i*k0/kp*d/dt+2*d^2/(dzdt)-d^2/dt^2)â = \chi*â
    :param k0p: k0/kp = central laser wavenumber (2pi/lambda_0) / plasma skin depth
    :param zmin: minimum value for zeta (nondimensionalized z-direction, zeta=k_p(z-ct))
    :param zmax: maximum value for zeta
    :param nz: number of grid points in zeta-direction
    :param dt: size of tau step (nondimensionalized time, tau=k_pct)
    :param nt: number of tau steps
    :param rmax: maximum value for rho (minimum value is always 0). rho = k_p*r (nondimensionalized radius)
    :param nr: number of rho steps
    :param a0: value for â(z,r,0), array of dimensions (nz, nr)
    :param aold: value for â(z,r,-1), array of dimensions (nz, nr)
    :return: a[z][r], 2D array of the value of â at every point zmin<=z<=zmax, 0<=r<rmax and t = dt*nt
    """
    # a_old corresponds to a(z, r, t-1), a_current corresponds to a(z, r, t) and a_new corresponds to a(z, r, t+1)
    a_old = np.zeros((nz + 2, nr), dtype=np.complex128)  # add 2 rows of ghost points in the zeta direction
    a_current = np.zeros((nz + 2, nr), dtype=np.complex128)
    a_new = np.zeros((nz + 2, nr), dtype=np.complex128)
    a_old[0:-2] = aold
    a_current[0:-2] = a0

    dz = (zmax - zmin) / (nz - 1)
    dr = rmax / (nr - 1)

    for t in range(0, nt):
        if t % 100 == 0:
            print("Time =", t * dt)
        # For every j, solve the tridiagonal system to calculate the solution over on the radius
        for jiter in prange(0, nz):
            j = nz - jiter
            d_upper = np.zeros(nr - 1, dtype=np.complex128)
            d_lower = np.zeros(nr - 1, dtype=np.complex128)
            d_main = np.zeros(nr, dtype=np.complex128)
            sol = np.zeros(nr, dtype=np.complex128)
            # boundary conditions for k = 0 and k = nr - 1
            d_main[0] = C(1, 0, k0p, dt, dz, dr) - chi(j - 1, 0, t) / 2 + 1j / dt * D(a_current, j - 1, 0, dz)
            d_main[-1] = (C(1, nr - 1, k0p, dt, dz, dr)
                          - chi(j - 1, nr - 1, t) / 2 + 1j / dt * D(a_current, j - 1, nr - 1, dz))
            sol[0] = (-2 / dt ** 2 * a_current[j - 1][0]
                      - (C(-1, 0, k0p, dt, dz, dr) - chi(j - 1, 0, t) - 1j / dt * D(a_current, j - 1, 0, dz))
                      * a_old[j - 1][0]
                      - L(1, 0, dr) / 2 * a_old[j - 1][1]
                      - 2 * np.exp(1j * (theta(a_current, j - 1, 0) - theta(a_current, j, 0))) / (dt * dz)
                      * (a_new[j][0] - a_old[j][0])
                      + np.exp(1j * (theta(a_current, j - 1, 0) - theta(a_current, j + 1, 0))) / (2 * dt * dz)
                      * (a_new[j + 1][0] - a_old[j + 1][0])
                      + testfunc(zmin + (j - 1) * dz, zmax))
            sol[-1] = (-2 / dt ** 2 * a_current[j - 1][-1]
                       - L(-1, nr - 1, dr) / 2 * a_old[j - 1][-2]
                       - (C(-1, nr - 1, k0p, dt, dz, dr) - chi(j - 1, nr - 1, t) - 1j / dt
                          * D(a_current, j - 1, nr - 1, dz))
                       * a_old[j - 1][-1]
                       - 2 * np.exp(1j * (theta(a_current, j - 1, nr - 1) - theta(a_current, j, nr - 1))) / (dt * dz)
                       * (a_new[j][-1] - a_old[j][-1])
                       + np.exp(1j * (theta(a_current, j - 1, nr - 1) - theta(a_current, j + 1, nr - 1)))
                       / (2 * dt * dz)
                       * (a_new[j + 1][-1] - a_old[j + 1][-1])
                       + testfunc(zmin + (j - 1) * dz, zmax))
            for k in prange(1, nr - 1):
                d_main[k] = C(1, k, k0p, dt, dz, dr) - chi(j - 1, k, t) / 2 + 1j / dt * D(a_current, j - 1, k, dz)
                sol[k] = (-2 / dt ** 2 * a_current[j - 1][k]
                          - L(-1, k, dr) / 2 * a_old[j - 1][k - 1]
                          - (C(-1, k, k0p, dt, dz, dr) - chi(j - 1, k, t) - 1j / dt * D(a_current, j - 1, k, dz))
                          * a_old[j - 1][k]
                          - L(1, k, dr) / 2 * a_old[j - 1][k + 1]
                          - 2 * np.exp(1j * (theta(a_current, j - 1, k) - theta(a_current, j, k))) / (dt * dz)
                          * (a_new[j][k] - a_old[j][k])
                          + np.exp(1j * (theta(a_current, j - 1, k) - theta(a_current, j + 1, k))) / (2 * dt * dz)
                          * (a_new[j + 1][k] - a_old[j + 1][k])
                          + testfunc(zmin + (j - 1) * dz, zmax))
                d_lower[k - 1] = L(-1, k, dr) / 2
                d_upper[k - 1] = L(1, k - 1, dr) / 2
            d_lower[-1] = L(-1, nr - 1, dr) / 2
            d_upper[-1] = L(1, nr - 2, dr) / 2
            a_new[j - 1] = TDMA(d_lower, d_main, d_upper, sol)
        a_old = a_current
        a_current = a_new
    return a_new
