import numpy as np

from scipy.integrate import quad

from multiprocessing import Pool, cpu_count

from numba import njit

from .tools import integral_trapezoid

### RSD theoretical calculation functions

@njit
def _omega_m(z, omegam_0):
    z = np.asarray(z)
    omegam_0 = np.asarray(omegam_0)
    return omegam_0 * (1 + z)**3 / (omegam_0 * (1 + z)**3 + 1 - omegam_0)

@njit
def _g(z, omegam_0, omegal_0=0.7):
    omz = _omega_m(z, omegam_0)
    olz = 1 - omz
    return 2.5 * omz / (omz**(4/7) - olz + (1 + omz/2)*(1 + olz/70))

@njit
def _D(z, omegam_0, omegal_0=0.7):
    return _g(z, omegam_0, omegal_0) / _g(0, omegam_0, omegal_0) / (1 + np.asarray(z))

@njit
def _sigma8(z, omegam_0, sigma8_0):
    return sigma8_0 * _D(z, omegam_0)

@njit
def _f(z, omegam_0, gamma):
    omega_m_z = _omega_m(z, omegam_0)
    return omega_m_z ** gamma

@njit
def fs8(z, gamma, omegam_0, sigma8_0):
    """Returns linear growth rate according to the redshift.

    Args:
        z (float): redshift
        gamma (float): γ
        omegam_0 (float): Ωm(0)
        sigma8_0 (float): σ8(0)

    Returns:
        float: fσ8(z)
    """
    return _f(z, omegam_0, gamma) * _sigma8(z, omegam_0, sigma8_0)

### SN1A theoretical calculation functions

@njit
def _H_LCDM(z, H0, omega_m):
    return H0 * np.sqrt(omega_m * (1+z)**3 + (1 - omega_m))

@njit
def _inv_H_LCDM(z, H0, omega_m):
    return 1.0 / _H_LCDM(z, H0, omega_m)

@njit
def _dL(z, H0, omega_m, c):
    integral = integral_trapezoid(_inv_H_LCDM, 0.0, z, 100, H0=H0, omega_m=omega_m)
    return (1 + z) * c * integral

@njit
def mu(z, omega_m, H0, c):
    return 5 * np.log10(_dL(z, H0, omega_m, c)) + 25

### BAO theoretical calculation functions

@njit
def _Dmrd(z, omega_m, rd, H0, c):
    integral = integral_trapezoid(_inv_H_LCDM, 0.0, z, 100, H0=H0, omega_m=omega_m)
    return c * integral / rd

@njit
def Dmrd_array(z_array, omega_m, rd, H0, c):
    result = np.empty_like(z_array)
    for i in range(z_array.size):
        result[i] = _Dmrd(z_array[i], omega_m, rd, H0, c)
    return result
