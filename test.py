from envelope_solver import *
import matplotlib.pyplot as plt

k0p = 100
zmin = -10
zmax = 5
nz = 100
dt = 1
nt = 1500
amp = 1.5
L = 1

Z = np.linspace(zmin, zmax, nz)
a0 = amp * np.sqrt(np.exp(-Z ** 2 / (2 * L ** 2)))

a = solve_1d(k0p, zmin, zmax, nz, dt, nt, a0)

plt.plot(Z, abs(a[0:-2]))
plt.show()
