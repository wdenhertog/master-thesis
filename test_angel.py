import numpy as np
import matplotlib.pyplot as plt
from laser_profiles import GaussianLaser
import aptools.plasma_accel.general_equations as ge
import scipy.constants as ct
from envelope_solver import *
from envelope_solver_2 import solve_2d as solve_2d_angel


# plasma parameters
n_p = 1e18 # cm^-3
s_d = ge.plasma_skin_depth(n_p)
k_p = 1 / s_d
w_p = ge.plasma_frequency(n_p)

# laser parameters in SI units
tau = 25e-15  # s
w_0 = 20e-6  # m
l_0 = 0.8e-6/2  # m
z_c = 0.  # m
a_0 = 1.
k_0 = 2*np.pi / l_0
k0p = k_0 / k_p
zR = ge.laser_rayleigh_length(w_0, l_0)
zR_norm = zR / s_d
print('Normalized Rayleigh length = {}'.format(zR_norm))

# time
dt = 10
t_max = 500
nt = int(t_max/dt)

# create laser pulse
laser_fb = GaussianLaser(a_0, w_0, tau, z_c, zf=0., lambda0=l_0)

# grid
zmin = -3
zmax = 3
nz = 400
rmax = 40
nr = 200
Z = np.linspace(zmin, zmax, nz)
R = np.linspace(0, rmax, nr)
ZZ, RR = np.meshgrid(Z, R)

# Get analytic laser envelope at t=0 and t=-1 (initial condition) and t=t_final
# (for comparison with numerical result)
a_2d0 = laser_fb.a_field(RR*s_d, 0, ZZ*s_d, 0)
a_2dm1 = laser_fb.a_field(RR*s_d, 0, ZZ*s_d - dt/w_p*ct.c, -dt/w_p)
a_2df = laser_fb.a_field(RR*s_d, 0, ZZ*s_d + t_max/w_p*ct.c, t_max/w_p)

# Print sum of initial a (kind of the integral)
print(np.sum(np.abs(a_2d0)))

# Plot initial pulse
plt.subplot(311)
plt.imshow(np.abs(a_2d0), aspect='auto', origin='lower')

# Solve and plot final pulse.
a2d = solve_2d(k0p, zmin, zmax, nz, dt, nt, rmax, nr, a_2d0.T, a_2dm1.T)
# a2d = solve_2d_angel(a_2dm1.T, a_2d0.T, nz, nt, nr, zmax, zmin, rmax, dt, k0p)
plt.subplot(312)
plt.imshow(np.abs(a2d[0:-2].T), aspect='auto', origin='lower')

# Print sum of final a (numerical)
print(np.sum(np.abs(a2d[0:-2])))

# Plot final pulse (analitical)
plt.subplot(313)
plt.imshow(np.abs(a_2df), aspect='auto', origin='lower')
# Print sum of final a (analytical)
print(np.sum(np.abs(a_2df)))

# Compare central slice
plt.figure()
plt.plot(np.abs(a_2d0)[:, int(nz/2)], label='initial (analytical)')
plt.plot(np.abs(a_2df)[:, int(nz/2)], label='final (analytical)')
plt.plot(np.abs(a2d)[0:-2].T[:, int(nz/2)], label='final (numerical)')
plt.xlabel('r [arb. u.]')
plt.ylabel('a')
plt.legend()
plt.show()
