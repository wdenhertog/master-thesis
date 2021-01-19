from wake_t.driver_witness import LaserPulse
import aptools.plasma_accel.general_equations as ge
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as ct
from envelope_solver import solve_2d
import time

n_p = 8e17  # cm^-3
s_d = ge.plasma_skin_depth(n_p)
k_p = 1 / s_d
w_p = ge.plasma_frequency(n_p)

# laser parameters in SI units
l_0 = 8e-6  # m
omega_0 = 2 * np.pi * ct.c / l_0
w_0 = 40e-6  # m
tau = 30e-15  # s
z_c = 0.  # m
a_0 = 2.96
k_0 = 2 * np.pi / l_0
k0p = k_0 / k_p
zR = ge.laser_rayleigh_length(w_0, l_0)
zR_norm = zR / s_d

# create laser pulse
laser = LaserPulse(z_c, l_0=l_0, w_0=w_0, a_0=a_0, tau=tau, polarization='linear')

# grid
t_old = -1 / w_p  # calculate solution at tau = -1
dist_to_focus = t_old * ct.c
dz = 0.8 * ct.c * k_p / omega_0
zmin = -544*dz
zmax = 544*dz
lrms = k_p * tau * ct.c
nz = int((zmax - zmin) / dz)
dr = 3.5 * ct.c * k_p / omega_0
rmax = 448*dr
nr = int(rmax / dr)
Z = np.linspace(zmin, zmax, nz)
R = np.linspace(0, rmax, nr)
ZZ, RR = np.meshgrid(Z, R)

a_2d0 = laser.get_a0_profile(RR * s_d, ZZ * s_d, 0)
a_old = laser.get_a0_profile(RR * s_d, ZZ * s_d, dist_to_focus)

dt = 0.9*dz

endtime = 1000
nt = int(endtime / dt)

dist_to_focus_end = dt * nt / w_p * ct.c

a_end_theoretical = laser.get_a0_profile(RR * s_d, ZZ * s_d, dist_to_focus_end)

start_time = time.time()
a2d = solve_2d(k0p, zmin, zmax, nz, dt, nt, rmax, nr, a_2d0.T, a_old.T)
print("--- %s seconds ---" % (time.time() - start_time))
suminit = np.sum(np.abs(a_2d0))
sumfinal = np.sum(np.abs(2*a2d))
sumdiff = abs(suminit - sumfinal)
sumdiffrel = sumdiff / suminit
print("Sum of initial values: {:.3f}, sum of final values: {:.3f}, diff: {:.3f},"
      " relative diff: {:.3f}".format(suminit, sumfinal, sumdiff, sumdiffrel))

sumend_theoretical = np.sum(np.abs(a_end_theoretical))
sumdiff = abs(sumend_theoretical - sumfinal)
sumdiffrel = sumdiff / sumend_theoretical
print("Sum of theoretical end values: {:.3f}, sum of end values: {:.3f}, diff: {:.3f},"
      " relative diff: {:.3f}".format(sumend_theoretical, sumfinal, sumdiff, sumdiffrel))

fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')
ax.plot_surface(ZZ, RR, a_old)
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
