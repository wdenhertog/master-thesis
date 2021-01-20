import matplotlib.pyplot as plt
from laser_profiles import GaussianLaser
import aptools.plasma_accel.general_equations as ge
from envelope_solver import *
import scipy.constants as ct
import time

# plasma parameters
n_p = 1e18  # cm^-3
s_d = ge.plasma_skin_depth(n_p)
k_p = 1 / s_d
w_p = ge.plasma_frequency(n_p)

# laser parameters in SI units
tau = 25e-15  # s
w_0 = 20e-6  # m
l_0 = 0.8e-6 / 2  # m
z_c = 0.  # m
a_0 = 1.
k_0 = 2 * np.pi / l_0
k0p = k_0 / k_p
zR = ge.laser_rayleigh_length(w_0, l_0)
zR_norm = zR / s_d
print('Normalized Rayleigh length = {}'.format(zR_norm))

# create laser pulse
laser_fb = GaussianLaser(a_0, w_0, tau, z_c, zf=0., lambda0=l_0)

# grid
zmin = -10
zmax = 10
nz = 400
rmax = 40
nr = 200
Z = np.linspace(zmin, zmax, nz)
R = np.linspace(0, rmax, nr)
ZZ, RR = np.meshgrid(Z, R)

# time
dt = 1
t_max = 1000
nt = int(t_max / dt)

# Get analytic laser envelope at t=0 and t=-1 (initial condition) and t=t_final
# (for comparison with numerical result)
a_2d0 = laser_fb.a_field(RR * s_d, 0, ZZ * s_d, 0)
a_2dm1 = laser_fb.a_field(RR * s_d, 0, ZZ * s_d - dt / w_p * ct.c, -dt / w_p)
a_2df = laser_fb.a_field(RR * s_d, 0, ZZ * s_d + t_max / w_p * ct.c, t_max / w_p)

start_time = time.time()
a2d = solve_2d(k0p, zmin, zmax, nz, dt, nt, rmax, nr, a_2d0.T, a_2dm1.T)
print("--- %s seconds ---" % (time.time() - start_time))

sumend_theoretical = np.sum(np.abs(a_2df))
sumdiff = np.sum(np.abs(a_2df-a2d[0:-2].T))
sumdiffrel = sumdiff / sumend_theoretical
print("diff: {:.3f}, relative diff: {:.3f}".format(sumdiff, sumdiffrel))

fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')
ax.plot_surface(ZZ, RR, abs(a_2d0))
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

fig4 = plt.figure()
ax = fig4.add_subplot(111, projection='3d')
ax.plot_surface(ZZ, RR, abs(a_2df))
plt.title("Theoretical Final solution")
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
