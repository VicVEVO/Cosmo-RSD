import numpy as np

from scipy.integrate import quad

from multiprocessing import Pool, cpu_count

# from functools import partial

import constants

### RSD theoretical calculation functions

def omega_m(z, omega_0):
    z = np.asarray(z)
    omega_0 = np.asarray(omega_0)
    return omega_0 * (1 + z)**3 / (omega_0 * (1 + z)**3 + 1 - omega_0)

def g(z, omega_0, omega_l0=0.7):
    omz = omega_m(z, omega_0)
    olz = 1 - omz
    return 2.5 * omz / (omz**(4/7) - olz + (1 + omz/2)*(1 + olz/70))

def D(z, omega_0, omega_l0=0.7):
    return g(z, omega_0, omega_l0) / g(0, omega_0, omega_l0) / (1 + np.asarray(z))

def sigma_8(z, omega_0, sigma_8_0):
    return sigma_8_0 * D(z, omega_0)

def f(gamma, z, omega_0):
    omega_m_z = omega_m(z, omega_0)
    return omega_m_z ** gamma

def growth(z, gamma = constants.GAMMA, omega_0 = constants.OMEGA_0, sigma_8_0 = constants.SIGMA_8_0):
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

### Pantheon+ theoretical calculation functions

def H_LCDM(z, H0, omega_m):
    return H0 * np.sqrt(omega_m * (1+z)**3 + (1 - omega_m))

def dL(z, H0, omega_m):
    c = constants.C
    integral, _ = quad(lambda zp: 1.0 / H_LCDM(zp, H0, omega_m), 0, z)
    return (1 + z) * c * integral

def mu(z, omega, H0):
    return 5 * np.log10(dL(z, H0, omega)) + 25

### Chi2 functions

def chi2_rsd(omega, sigma, gamma):
    """ Returns chi2 value for a given omega, sigma, gamma.

    Args:
        omega (float)
        sigma (float)
        gamma (float)

    Returns:
        float: chi2
    """
    model = growth(constants.z_data, gamma=gamma, omega_0=omega, sigma_8_0=sigma)

    errors = np.asarray(0.5 * (constants.fs8_err_plus + constants.fs8_err_minus))
    residuals = (model - constants.fs8_data.values) / errors
    return np.sum(residuals**2)

def chi2_panth(omega, H0, M):
    """ Returns chi2 value for a given omega, H0, M.

    Args:
        omega (float)
        H0 (float)
        M (float)

    Returns:
        float: chi2
    """
    delta_mu = np.empty(constants.n_panth)

    for i in range(constants.n_panth):
        if constants.is_calibrator_panth[i] == 0:
            delta_mu[i] = mu(constants.z_data_panth[i], omega, H0) - (constants.m_b_corr_panth[i] - M)
        else:
            delta_mu[i] = constants.m_b_corr_panth[i] - M - constants.ceph_dist_panth[i]

    return delta_mu @ constants.inv_cov_panth @ delta_mu

def chi2_rsd_panth(omega, sigma, gamma, H0, M):
    return chi2_rsd(omega, sigma, gamma) + chi2_panth(omega, H0, M)

# def chi2_rsd_3D_version(omega_vals, sigma_vals, gamma_vals):
#     """ Returns chi2 value for all sigma, gamma and omega values.
#     Parallelized function.
#     Unused here.

#     Args:
#         omega_vals (tab): all values of omega
#         sigma_vals (tab): all values of sigma
#         gamma_vals (tab): all values of gamma

#     Returns:
#         3D table
#     """
#     n_omega = len(omega_vals)
#     n_sigma = len(sigma_vals)
#     n_gamma = len(gamma_vals)

#     z_data = np.asarray(constants.z_data.values)
#     fs8_data = np.asarray(constants.fs8_data.values)
#     errors = np.asarray(0.5 * (constants.fs8_err_plus + constants.fs8_err_minus))

#     chi2_array = np.empty((n_omega, n_sigma, n_gamma))

#     tasks = [(i, j) for i in range(n_omega) for j in range(n_sigma)]

#     func = partial(
#         compute_chi2_for_omega_sigma,
#         omega_vals=omega_vals,
#         sigma_vals=sigma_vals,
#         gamma_vals=gamma_vals,
#         z_data=z_data,
#         fs8_data=fs8_data,
#         errors=errors
#     )

#     with Pool(cpu_count()) as pool:
#         results = pool.starmap(func, tasks)

#     for i, j, chi2_vals in results:
#         chi2_array[i, j, :] = chi2_vals

#     return chi2_array