from envelope_solver import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

k0p = 100
zmin = -5
zmax = 5
nz = 1000
dt = 1
nt = 10
rmax = 5
nr = 100
a0 = 1
sz = np.sqrt(2)
sr = np.sqrt(2)

Z = np.linspace(zmin, zmax, nz)
R = np.linspace(0, rmax, nr)
ZZ, RR = np.meshgrid(R, Z)
a_env0 = a0 / np.sqrt(2) * np.sqrt(np.exp(-Z ** 2 / (sz ** 2)))
a_2d0 = np.zeros((nz, nr))
for i in range(nz):
    for j in range(nr):
        a_2d0[i][j] = a0 / np.sqrt(2) * np.sqrt(np.exp(-Z[i] ** 2 / (sz ** 2) - R[j] ** 2 / (sr ** 2)))

# a = solve_1d(k0p, zmin, zmax, nz, dt, nt, a_env0)
# a2d3 = solve_2d(k0p, zmin, zmax, nz, dt, 3, rmax, nr, a_2d0)
# a2d5 = solve_2d(k0p, zmin, zmax, nz, dt, 5, rmax, nr, a_2d0)
a2d = solve_2d(k0p, zmin, zmax, nz, dt, nt, rmax, nr, a_2d0)

# plt.plot(Z, a_env0)
# plt.plot(Z, abs(a2d[0:-2, 0]))
# plt.plot(Z, abs(a_2d0[:, 9]))
# plt.show()
fig1 = plt.figure()
ha = fig1.add_subplot(111, projection='3d')
ha.plot_surface(ZZ, RR, a_2d0)

fig2 = plt.figure()
hb = fig2.add_subplot(111, projection='3d')
hb.plot_surface(ZZ, RR, abs(a2d[0:-2, :]))

plt.show()
