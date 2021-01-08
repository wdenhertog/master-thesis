from envelope_solver import solve_2d_test
import numpy as np
import matplotlib.pyplot as plt


def a(z, r,zm, rm):
    return (-r ** 2 + rm ** 2) * (z-zm)**2


zmin = -10
zmax = 10
nz = 200
dz = (zmax-zmin)/(nz-1)
rmax = 25
nr = 50
Z = np.linspace(zmin, zmax, nz)
Z_copy = np.linspace(zmin, zmax+2*dz, nz+2)
R = np.linspace(0, rmax, nr)
ZZ, RR = np.meshgrid(Z, R)
a_init = np.zeros((nr, nz))
ZZ_copy, RR_copy = np.meshgrid(Z_copy, R)
for i in range(nz):
    for j in range(nr):
        a_init[j, i] = (zmin + i * dz) * rmax ** 2

dt = 1
nt = 1000

k0p = 1

func = a(ZZ, RR, zmax, rmax)
a_init = func
a_old = a_init

a_test = solve_2d_test(k0p, zmin, zmax, nz, dt, nt, rmax, nr, a_init.T, a_old.T)
a_test_copy = a_test
a_test = a_test[0:-2, :].T
print(np.linalg.norm(abs(a_test) - abs(func)))

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

