from envelope_solver import solve_2d_test
import numpy as np
import matplotlib.pyplot as plt


def a(z, r, zm, rm):
    return (-r ** 2 + rm ** 2) * (z - zm)


zmin = -10
zmax = 10
nz = 40
dz = (zmax - zmin) / (nz - 1)

rmax = 25
nr = 400
dr = rmax / (nr - 1)

Z = np.linspace(zmin, zmax, nz)
R = np.linspace(0, rmax, nr)
ZZ, RR = np.meshgrid(Z, R)

a_init = np.ones((nr, nz))
a_old = np.copy(a_init)

dt = 1000
nt = 2000

k0p = 100

func = a(ZZ, RR, zmax, rmax)

a_test = solve_2d_test(k0p, zmin, zmax, nz, rmax, nr, dt, nt, a_init.T, a_old.T)
a_test_copy = a_test
a_test = a_test[0:-2, :].T
suminit = np.sum(np.abs(func))
sumdiff = np.sum(abs(func - a_test))
sumdiffrel = sumdiff / suminit
print("diff: {:.3f}, relative diff: {:.3f}".format(sumdiff, sumdiffrel))

fig2 = plt.figure()
ax = fig2.add_subplot(111, projection='3d')
ax.plot_surface(ZZ, RR, np.abs(func))
plt.title("Target function")
plt.xlabel("$\\zeta$")
plt.ylabel("$\\rho$")
ax.set_zlabel("$\\|\hat{a}|$")

fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')
ax.plot_surface(ZZ, RR, np.abs(a_test))
plt.title("Generated function")
plt.xlabel("$\\zeta$")
plt.ylabel("$\\rho$")
ax.set_zlabel("$\\|\hat{a}|$")
