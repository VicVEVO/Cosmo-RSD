# f = d ln(D) / d ln(a) justine, christophe (frank), raidwan
import numpy as np
from astropy.cosmology import Planck18 as cosmo
import constants

# def S_8(sigma_8, omega_m):
#     return sigma_8 * np.sqrt(omega_m / 0.3)


def gamma(z):
    return GAMMA

def D(z):
    return 1 / (1 + z)

def sigma_8(z):
    return SIGMA_8_0 * D(z)

def f(z):
    """

    Args:
        z (float):

    Returns:
        float: f(z) = Ω_m(z)^γ(z)
    """
    return cosmo.Om(z) ** gamma(z)

def growth_rate(z):
    """

    Args:
        z (float):

    Returns:
        float: fσ₈(z)
    """
    return f(z) * sigma_8(z)

