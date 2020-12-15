from envelope_solver import *

k0p = 100
zmin = -5
zmax = 5
nz = 10
dt = 1
nt = 10
amp = 1.5
L = 1

Z = np.linspace(zmin, zmax, nz)
a0 = amp ** 2 * np.exp(-Z ** 2 / (2 * L ** 2))

a = solve_1d(k0p, zmin, zmax, nz, dt, nt, a0)
