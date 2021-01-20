# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines a set of common laser profiles.
"""
import numpy as np
from scipy.constants import c, m_e, e
from scipy.special import factorial, genlaguerre, binom

# Generic classes
# ---------------

class LaserProfile( object ):
    """
    Base class for all laser profiles.
    Any new laser profile should inherit from this class, and define its own
    `E_field` method, using the same signature as the method below.
    Profiles that inherit from this base class can be summed,
    using the overloaded + operator.
    """
    def __init__( self, propagation_direction ):
        """
        Initialize the propagation direction of the laser.
        (Each subclass should call this method at initialization.)
        Parameter
        ---------
        propagation_direction: int
            Indicates in which direction the laser propagates.
            This should be either 1 (laser propagates towards positive z)
            or -1 (laser propagates towards negative z).
        """
        assert propagation_direction in [-1, 1]
        self.propag_direction = float(propagation_direction)

    def E_field( self, x, y, z, t ):
        """
        Return the electric field of the laser
        Parameters
        -----------
        x, y, z: ndarrays (meters)
            The positions at which to calculate the profile (in the lab frame)
        t: ndarray or float (seconds)
            The time at which to calculate the profile (in the lab frame)
        Returns:
        --------
        Ex, Ey: ndarrays (V/m)
            Arrays of the same shape as x, y, z, containing the fields
        """
        # The base class only defines dummy fields
        # (This should be replaced by any class that inherits from this one.)
        return( np.zeros_like(x), np.zeros_like(x) )

    def a_field( self, x, y, z, t ):
        """
        Return the envelope of the (dimensionless) vector potential of the laser
        Parameters
        ----------
        x, y, z: ndarrays (meters)
            The positions at which to calculate the profile (in the lab frame)
        t: ndarray or float (seconds)
            The time at which to calculate the profile (in the lab frame)
        Returns
        -------
        a: ndarrays (adimensionnal)
            Array of the same shape as x, y, z, containing the fields
        """
        # The base class only defines dummy fields
        # (This should be replaced by any class that inherits from this one.)
        return( np.zeros_like(x) )

    def __add__( self, other ):
        """
        Overload the + operations for laser profiles
        """
        return( SummedLaserProfile( self, other ) )


# Particular classes for each laser profile
# -----------------------------------------

class GaussianLaser( LaserProfile ):
    """Class that calculates a Gaussian laser pulse."""

    def __init__( self, a0, waist, tau, z0, zf=None, theta_pol=0.,
                    lambda0=0.8e-6, cep_phase=0., phi2_chirp=0.,
                    propagation_direction=1 ):
        """
        Define a linearly-polarized Gaussian laser profile.
        More precisely, the electric field **near the focal plane**
        is given by:
        .. math::
            E(\\boldsymbol{x},t) = a_0\\times E_0\,
            \exp\left( -\\frac{r^2}{w_0^2} - \\frac{(z-z_0-ct)^2}{c^2\\tau^2} \\right)
            \cos[ k_0( z - z_0 - ct ) - \phi_{cep} ]
        where :math:`k_0 = 2\pi/\\lambda_0` is the wavevector and where
        :math:`E_0 = m_e c^2 k_0 / q_e` is the field amplitude for :math:`a_0=1`.
        .. note::
            The additional terms that arise **far from the focal plane**
            (Gouy phase, wavefront curvature, ...) are not included in the above
            formula for simplicity, but are of course taken into account by
            the code, when initializing the laser pulse away from the focal plane.
        Parameters
        ----------
        a0: float (dimensionless)
            The peak normalized vector potential at the focal plane, defined
            as :math:`a_0` in the above formula.
        waist: float (in meter)
            Laser waist at the focal plane, defined as :math:`w_0` in the
            above formula.
        tau: float (in second)
            The duration of the laser (in the lab frame),
            defined as :math:`\\tau` in the above formula.
        z0: float (in meter)
            The initial position of the centroid of the laser
            (in the lab frame), defined as :math:`z_0` in the above formula.
        zf: float (in meter), optional
            The position of the focal plane (in the lab frame).
            If ``zf`` is not provided, the code assumes that ``zf=z0``, i.e.
            that the laser pulse is at the focal plane initially.
        theta_pol: float (in radian), optional
           The angle of polarization with respect to the x axis.
        lambda0: float (in meter), optional
            The wavelength of the laser (in the lab frame), defined as
            :math:`\\lambda_0` in the above formula.
            Default: 0.8 microns (Ti:Sapph laser).
        cep_phase: float (in radian), optional
            The Carrier Enveloppe Phase (CEP), defined as :math:`\phi_{cep}`
            in the above formula (i.e. the phase of the laser
            oscillation, at the position where the laser enveloppe is maximum)
        phi2_chirp: float (in second^2)
            The amount of temporal chirp, at focus (in the lab frame)
            Namely, a wave packet centered on the frequency
            :math:`(\omega_0 + \delta \omega)` will reach its peak intensity
            at :math:`z(\delta \omega) = z_0 - c \phi^{(2)} \, \delta \omega`.
            Thus, a positive :math:`\phi^{(2)}` corresponds to positive chirp,
            i.e. red part of the spectrum in the front of the pulse and blue
            part of the spectrum in the back.
        propagation_direction: int, optional
            Indicates in which direction the laser propagates.
            This should be either 1 (laser propagates towards positive z)
            or -1 (laser propagates towards negative z).
        """
        # Initialize propagation direction
        LaserProfile.__init__(self, propagation_direction)

        # Set a number of parameters for the laser
        k0 = 2*np.pi/lambda0
        E0 = a0*m_e*c**2*k0/e
        zr = 0.5*k0*waist**2

        # If no focal plane position is given, use z0
        if zf is None:
            zf = z0

        # Store the parameters
        self.k0 = k0
        self.inv_zr = 1./zr
        self.zf = zf
        self.z0 = z0
        self.E0x = E0 * np.cos(theta_pol)
        self.E0y = E0 * np.sin(theta_pol)
        self.a0 = a0
        self.w0 = waist
        self.cep_phase = cep_phase
        self.phi2_chirp = phi2_chirp
        self.inv_ctau2 = 1./(c*tau)**2

    def E_field( self, x, y, z, t ):
        """
        See the docstring of LaserProfile.E_field
        """
        # Note: this formula is expressed with complex numbers for compactness
        # and simplicity, but only the real part is used in the end
        # (see final return statement)
        # The formula for the laser (in complex numbers) is obtained by
        # multiplying the Fourier transform of the laser at focus
        # E(k_x,k_y,\omega) = exp( -(\omega-\omega_0)^2(\tau^2/4 + \phi^(2)/2)
        # - (k_x^2 + k_y^2)w_0^2/4 ) by the paraxial propagator
        # e^(-i(\omega/c - (k_x^2 +k_y^2)/2k0)(z-z_foc))
        # and then by taking the inverse Fourier transform in x, y, and t

        # Diffraction and stretch_factor
        prop_dir = self.propag_direction
        diffract_factor = 1. + 1j * prop_dir*(z - self.zf) * self.inv_zr
        stretch_factor = 1 - 2j * self.phi2_chirp * c**2 * self.inv_ctau2
        # Calculate the argument of the complex exponential
        exp_argument = - 1j*self.cep_phase \
            + 1j*self.k0*( prop_dir*(z - self.z0) - c*t ) \
            - (x**2 + y**2) / (self.w0**2 * diffract_factor) \
            - 1./stretch_factor*self.inv_ctau2 * \
                                    ( prop_dir*(z - self.z0) - c*t )**2
        # Get the transverse profile
        profile = np.exp(exp_argument) /(diffract_factor * stretch_factor**0.5)

        # Get the projection along x and y, with the correct polarization
        Ex = self.E0x * profile
        Ey = self.E0y * profile

        return( Ex.real, Ey.real )

    def a_field(self, x, y, z, t):
        """
        See the docstring of LaserProfile.a_field
        """
        diffract_factor = 1. + 1j * ( z - self.zf ) * self.inv_zr
        stretch_factor = 1 - 2j * self.phi2_chirp * c**2 * self.inv_ctau2
        # Calculate the argument of the complex exponential
        exp_argument = -1j*self.cep_phase  \
            - (x**2 + y**2) / (self.w0**2 * diffract_factor) \
            - 1./stretch_factor * self.inv_ctau2 * ( z - self.z0 - c*t)**2
        # Get the transverse profile
        profile = np.exp(exp_argument) / ( diffract_factor * stretch_factor**0.5 )
        a = self.a0 * profile
        return a