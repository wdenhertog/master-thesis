from envelope_solver import *
import matplotlib.pyplot as plt
from wake_t.driver_witness import LaserPulse

k0p = 20
zmin = -10
zmax = 10
nz = 100
dt = 1
nt = 1500
rmax = 5
nr = 30
a0 = 1.5
sz = np.sqrt(2)
sr = 1

Z = np.linspace(zmin, zmax, nz)
R = np.linspace(0, rmax, nr)
ZZ, RR = np.meshgrid(Z, R)

a_2d0 = a0 / np.sqrt(2) * np.sqrt(np.exp(-ZZ ** 2 / (sz ** 2) - RR ** 2 / (sr ** 2)))

laser = LaserPulse(0, l_0=1, w_0=2, a_0=1, tau=1)
# a_2d0 = laser.get_a0_profile(ZZ, RR, np.sqrt(ZZ ** 2 + RR ** 2))  # uncomment this line for LaserPulse initial value

a2d = solve_2d(k0p, zmin, zmax, nz, dt, nt, rmax, nr, a_2d0.T)

fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')
ax.plot_surface(ZZ, RR, a_2d0)
plt.title("Initial value")
plt.xlabel("$\\zeta$")
plt.ylabel("$\\rho$")
ax.set_zlabel("$\\|\hat{a}|$")

fig2 = plt.figure()
ax = fig2.add_subplot(111, projection='3d')
ax.plot_surface(ZZ, RR, abs(a2d[0:-2, :].T))
plt.title("Final solution")
plt.xlabel("$\\zeta$")
plt.ylabel("$\\rho$")
ax.set_zlabel("$\\|\hat{a}|$")

fig3 = plt.figure(figsize=(10, 9))
ax = fig3.add_subplot(221)
ax.plot(Z, abs(a2d[0:-2, 0].T))
plt.title("Final solution at r=0")
plt.xlabel("$\\zeta$")
plt.ylabel("$\\|\hat{a}|$")

ax = fig3.add_subplot(222)
ax.plot(Z, abs(a2d[0:-2, int(0.25 * nr)].T))
plt.title("Final solution at r=0.25*rmax")
plt.xlabel("$\\zeta$")
plt.ylabel("$\\|\hat{a}|$")

ax = fig3.add_subplot(223)
ax.plot(Z, abs(a2d[0:-2, int(0.5 * nr)].T))
plt.title("Final solution at r=0.5*rmax")
plt.xlabel("$\\zeta$")
plt.ylabel("$\\|\hat{a}|$")

ax = fig3.add_subplot(224)
ax.plot(Z, abs(a2d[0:-2, int(0.75 * nr)].T))
plt.title("Final solution at r=0.75*rmax")
plt.xlabel("$\\zeta$")
plt.ylabel("$\\|\hat{a}|$")

plt.show()
