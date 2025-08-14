import numpy as np

from scipy.integrate import quad

from multiprocessing import Pool, cpu_count

from numba import njit

from .tools import integral_trapezoid

### RSD theoretical calculation functions

@njit
def _Omega_m(z, Omega_m_0):
    z = np.asarray(z)
    Omega_m_0 = np.asarray(Omega_m_0)
    return Omega_m_0 * (1 + z)**3 / (Omega_m_0 * (1 + z)**3 + 1 - Omega_m_0)

@njit
def _g(z, Omega_m_0, Omega_l_0=0.7):
    omz = _Omega_m(z, Omega_m_0)
    olz = 1 - omz
    return 2.5 * omz / (omz**(4/7) - olz + (1 + omz/2)*(1 + olz/70))

@njit
def _D(z, Omega_m_0, Omega_l_0=0.7):
    return _g(z, Omega_m_0, Omega_l_0) / _g(0, Omega_m_0, Omega_l_0) / (1 + np.asarray(z))

@njit
def _sigma8(z, Omega_m_0, sigma_8_0):
    return sigma_8_0 * _D(z, Omega_m_0)

@njit
def _f(z, Omega_m_0, gamma):
    Omega_m_z = _Omega_m(z, Omega_m_0)
    return Omega_m_z ** gamma

@njit
def fs8(z, gamma, Omega_m_0, sigma_8_0):
    """Returns linear growth rate according to the redshift.

    Args:
        z (float): redshift
        gamma (float): γ
        Omega_m_0 (float): Ωm(0)
        sigma_8_0 (float): σ8(0)

    Returns:
        float: fσ8(z)
    """
    return _f(z, Omega_m_0, gamma) * _sigma8(z, Omega_m_0, sigma_8_0)

### SN1A theoretical calculation functions

@njit
def _H_LCDM(z, H0, Omega_m_0):
    return H0 * np.sqrt(Omega_m_0 * (1+z)**3 + (1 - Omega_m_0))

@njit
def _inv_H_LCDM(z, H0, Omega_m_0):
    return 1.0 / _H_LCDM(z, H0, Omega_m_0)

@njit
def _dL(z, H0, Omega_m_0, c):
    integral = integral_trapezoid(_inv_H_LCDM, 0.0, z, 100, H0=H0, Omega_m_0=Omega_m_0)
    return (1 + z) * c * integral

@njit
def mu(z, Omega_m_0, H0, c):
    return 5 * np.log10(_dL(z, H0, Omega_m_0, c)) + 25

### BAO theoretical calculation functions

@njit
def _Dmrd(z, Omega_m_0, rd, H0, c):
    integral = integral_trapezoid(_inv_H_LCDM, 0.0, z, 100, H0=H0, Omega_m_0=Omega_m_0)
    return c * integral / rd

@njit
def Dmrd_array(z_array, Omega_m_0, rd, H0, c):
    result = np.empty_like(z_array)
    for i in range(z_array.size):
        result[i] = _Dmrd(z_array[i], Omega_m_0, rd, H0, c)
    return result
