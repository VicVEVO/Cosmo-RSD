import numpy as np

from scipy.integrate import quad

from multiprocessing import Pool, cpu_count

from numba import njit

from . import constants

@njit
def integral_trapezoid(func, a, b, N, H0, omega_m):
    h = (b - a) / N
    result = 0.5 * (func(a, H0, omega_m) + func(b, H0, omega_m))
    for i in range(1, N):
        result += func(a + i * h, H0, omega_m)
    result *= h
    return result

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
def _fs8(z, gamma, omegam_0, sigma8_0):
    """Returns linear growth rate according to the redshift.

    Args:
        z (float): redshift
        gamma (float): γ. Defaults to constants.GAMMA.
        omegam_0 (float): Ωm(0). Defaults to constants.omegam_0.
        sigma8_0 (float): σ8(0). Defaults to constants.sigma8_0.

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
    integral = integral_trapezoid(_inv_H_LCDM, 0.0, z, 100, H0, omega_m)
    return (1 + z) * c * integral

@njit
def _mu(z, omega_m, H0, c):
    return 5 * np.log10(_dL(z, H0, omega_m, c)) + 25

### BAO theoretical calculation functions

@njit
def _Dmrd(z, omega_m, rd, H0, c):
    integral = integral_trapezoid(_inv_H_LCDM, 0.0, z, 1000, H0, omega_m)
    return c * integral / rd

@njit
def _Dmrd_array(z_array, omega_m, rd, H0, c):
    result = np.empty_like(z_array)
    for i in range(z_array.size):
        result[i] = _Dmrd(z_array[i], omega_m, rd, H0, c)
    return result

### Chi2 functions

@njit
def chi2_rsd(z_data, fs8_data, fs8_err_plus, fs8_err_minus, omega_m, sigma_8, gamma):
    """ Returns chi2 value for a given Omega_m, sigma_8, gamma.

    Args:
        z_data (array)
        fs8_data (array): constants.fs8_data.values
        fs8_err_plus (array)
        fs8_err_minus (array)
        omega_m (float)
        sigma_8 (float)
        gamma (float)

    Returns:
        float: chi2
    """
    model = _fs8(z_data, gamma=gamma, omegam_0=omega_m, sigma8_0=sigma_8)

    errors = np.asarray(0.5 * (fs8_err_plus + fs8_err_minus))
    residuals = (model - fs8_data) / errors
    return np.sum(residuals**2)

@njit
def chi2_panth(n_panth, z_data_panth, is_calibrator_panth, m_b_corr_panth, ceph_dist_panth, inv_cov_panth, omega_m, H0, M, c):
    """Returns chi2 value for a given Omega_m, H0, M according to Pantheon+ data.

    Args:
        n_panth (int)
        z_data_panth (array)
        is_calibrator_panth (bool)
        m_b_corr_panth (array)
        ceph_dist_panth (array)
        inv_cov_panth (array)
        omega_m (float)
        H0 (float)
        M (float)
        c (float)

    Returns:
        float: chi2
    """
    delta_mu = np.empty(n_panth)

    for i in range(n_panth):
        if is_calibrator_panth[i] == 0:
            delta_mu[i] = _mu(z_data_panth[i], omega_m, H0, c) - (m_b_corr_panth[i] - M)
        else:
            delta_mu[i] = m_b_corr_panth[i] - M - ceph_dist_panth[i]

    return delta_mu @ inv_cov_panth @ delta_mu

@njit
def chi2_bao_dmrd(z_data, dmrd_data, dmrd_err, c, omega_m, rd, H0):
    """Returns chi2 value for a given omega, rd, H0 according to dm/rd (from BAO) data.

    Args:
        z_data (array)
        dmrd_data (array)
        dmrd_err (array)
        c (float)
        omega_m (float)
        rd (float)
        H0 (float)

    Returns:
        float: chi2
    """
    model = _Dmrd_array(z_data, omega_m, rd, H0, c)
    residuals = (model - dmrd_data) / dmrd_err
    return np.sum(residuals**2)
