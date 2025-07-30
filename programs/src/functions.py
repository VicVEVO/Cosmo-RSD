import numpy as np

from scipy.integrate import quad

from multiprocessing import Pool, cpu_count

from numba import njit

import constants

### RSD theoretical calculation functions

@njit
def omega_m(z, omega_0):
    z = np.asarray(z)
    omega_0 = np.asarray(omega_0)
    return omega_0 * (1 + z)**3 / (omega_0 * (1 + z)**3 + 1 - omega_0)

@njit
def g(z, omega_0, omega_l0=0.7):
    omz = omega_m(z, omega_0)
    olz = 1 - omz
    return 2.5 * omz / (omz**(4/7) - olz + (1 + omz/2)*(1 + olz/70))

@njit
def D(z, omega_0, omega_l0=0.7):
    return g(z, omega_0, omega_l0) / g(0, omega_0, omega_l0) / (1 + np.asarray(z))

@njit
def sigma_8(z, omega_0, sigma_8_0):
    return sigma_8_0 * D(z, omega_0)

@njit
def f(gamma, z, omega_0):
    omega_m_z = omega_m(z, omega_0)
    return omega_m_z ** gamma

@njit
def growth(z, gamma, omega_0, sigma_8_0):
    """Returns linear growth rate according to the redshift.

    Args:
        z (float): redshift
        gamma (float): γ. Defaults to constants.GAMMA.
        omega_0 (float): Ωm(0). Defaults to constants.OMEGA_0.
        sigma_8_0 (float): σ8(0). Defaults to constants.SIGMA_8_0.

    Returns:
        float: fσ8(z)
    """
    return f(gamma, z, omega_0) * sigma_8(z, omega_0, sigma_8_0)

### SN1A theoretical calculation functions

@njit
def H_LCDM(z, H0, omega_m):
    return H0 * np.sqrt(omega_m * (1+z)**3 + (1 - omega_m))


@njit
def integral_trapezoid(func, a, b, N, H0, omega_m):
    h = (b - a) / N
    result = 0.5 * (func(a, H0, omega_m) + func(b, H0, omega_m))
    for i in range(1, N):
        result += func(a + i * h, H0, omega_m)
    result *= h
    return result

@njit
def inv_H_LCDM(z, H0, omega_m):
    return 1.0 / H_LCDM(z, H0, omega_m)

@njit
def dL(z, H0, omega_m, c):
    integral = integral_trapezoid(inv_H_LCDM, 0.0, z, 100, H0, omega_m)
    return (1 + z) * c * integral

# @njit
def ex_dL(z, H0, omega_m):
    c = constants.C
    integral, _ = quad(lambda zp: 1.0 / H_LCDM(zp, H0, omega_m), 0, z)
    return (1 + z) * c * integral

@njit
def mu(z, omega, H0, c):
    return 5 * np.log10(dL(z, H0, omega, c)) + 25

### BAO theoretical calculation functions

@njit
def Dmrd(z, omega_m, rd, H0, c):
    integral = integral_trapezoid(inv_H_LCDM, 0.0, z, 1000, H0, omega_m)
    return c * integral / rd

@njit
def Dmrd_array(z_array, omega_m, rd, H0, c):
    result = np.empty_like(z_array)
    for i in range(z_array.size):
        result[i] = Dmrd(z_array[i], omega_m, rd, H0, c)
    return result

### Chi2 functions

@njit
def chi2_rsd(z_data, fs8_data, fs8_err_plus, fs8_err_minus, omega, sigma, gamma):
    """ Returns chi2 value for a given omega, sigma, gamma.

    Args:
        z_data
        fs8_data: constants.fs8_data.values
        fs8_err_plus
        fs8_err_minus
        omega (float)
        sigma (float)
        gamma (float)

    Returns:
        float: chi2
    """
    model = growth(z_data, gamma=gamma, omega_0=omega, sigma_8_0=sigma)

    errors = np.asarray(0.5 * (fs8_err_plus + fs8_err_minus))
    residuals = (model - fs8_data) / errors
    return np.sum(residuals**2)

@njit
def chi2_panth(n_panth, z_data_panth, is_calibrator_panth, m_b_corr_panth, ceph_dist_panth, inv_cov_panth, omega, H0, M, c):
    """ Returns chi2 value for a given omega, H0, M according to Pantheon+ data.

    Args:
        omega (float)
        H0 (float)
        M (float)

    Returns:
        float: chi2
    """
    delta_mu = np.empty(n_panth)

    for i in range(n_panth):
        if is_calibrator_panth[i] == 0:
            delta_mu[i] = mu(z_data_panth[i], omega, H0, c) - (m_b_corr_panth[i] - M)
        else:
            delta_mu[i] = m_b_corr_panth[i] - M - ceph_dist_panth[i]

    return delta_mu @ inv_cov_panth @ delta_mu

@njit
def chi2_bao_dmrd(z_data, dmrd_data, dmrd_err, c, omega, rd, H0):
    model = Dmrd_array(z_data, omega, rd, H0, c)
    residuals = (model - dmrd_data) / dmrd_err
    return np.sum(residuals**2)

@njit
def chi2_rsd_panth(z_data, fs8_data, fs8_err_plus, fs8_err_minus, n_panth, z_data_panth, is_calibrator_panth, m_b_corr_panth, ceph_dist_panth, inv_cov_panth, omega, sigma, gamma, H0, M, c):
    return chi2_rsd(z_data, fs8_data, fs8_err_plus, fs8_err_minus, omega, sigma, gamma) + chi2_panth(n_panth, z_data_panth, is_calibrator_panth, m_b_corr_panth, ceph_dist_panth, inv_cov_panth, omega, H0, M, c)

