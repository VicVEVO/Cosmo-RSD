import numpy as np

import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches

from scipy.stats import gaussian_kde

from numpy.polynomial.chebyshev import Chebyshev
from rich.console import Console
from rich.table import Table

from multiprocessing import Pool, cpu_count

import fastkde

import constants

### Theoretical calculation functions

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

### Chi2

def chi2(omega_vals, sigma_vals, gamma_vals):

    n_omega = len(omega_vals)
    n_sigma = len(sigma_vals)
    n_gamma = len(gamma_vals)

    z_data = np.asarray(constants.z_data.values)
    fs8_data = np.asarray(constants.fs8_data.values)
    errors = np.asarray(0.5 * (constants.fs8_err_plus + constants.fs8_err_minus))

    chi2_array = np.empty((n_omega, n_sigma, n_gamma))

    for i, omega in enumerate(omega_vals):
        for j, sigma in enumerate(sigma_vals):
            z_broadcast = z_data[None, :]
            gamma_broadcast = gamma_vals[:, None]

            model = growth(z_broadcast, gamma=gamma_broadcast, omega_0=omega, sigma_8_0=sigma)

            residuals = (model - fs8_data[None, :]) / errors[None, :]
            chi2_vals = np.sum(residuals**2, axis=1)

            chi2_array[i, j, :] = chi2_vals

    return chi2_array


def calc_chi2_gamma_sigma(sigma_8, gamma_value):
    gamma_func = make_constant_gamma_func(gamma_value)
    errors = 0.5 * (constants.fs8_err_plus + constants.fs8_err_minus)
    chi2 = np.sum((growth(constants.z_data.values, gamma_func, sigma_8_0 = sigma_8) - constants.fs8_data)**2 / errors**2)
    return chi2

def calc_chi2(omega0_val, sigma8_val, gamma_val):
    gamma_func = make_constant_gamma_func(gamma_val)
    errors = 0.5 * (constants.fs8_err_plus + constants.fs8_err_minus)
    chi2 = np.sum((growth(constants.z_data.values, gamma_func, omega_0=omega0_val, sigma_8_0 = sigma8_val) - constants.fs8_data)**2 / errors**2)
    return chi2

def calc_chi2_gamma_omega(omega_0, gamma_value):
    gamma_func = make_constant_gamma_func(gamma_value)
    errors = 0.5 * (constants.fs8_err_plus + constants.fs8_err_minus)
    chi2 = np.sum((growth(constants.z_data.values, gamma_func, omega_0) - constants.fs8_data)**2 / errors**2)
    return chi2

def calc_chi2_sigma_omega_vectorized2(omega0, sigma8_0):
    omega0 = np.asarray(omega0)
    sigma8_0 = np.asarray(sigma8_0)
    errors = 0.5 * (constants.fs8_err_plus + constants.fs8_err_minus)
    
    model = growth(constants.z_data.values, omega_0=omega0[..., None], sigma_8_0=sigma8_0[..., None])
    residuals = (model - constants.fs8_data) / errors
    chi2 = np.sum(residuals**2)
    return chi2

def calc_chi2_sigma_omega_vectorized(omega0, sigma8_0, n_gamma=600):
    gamma_vals = np.linspace(0, 1, n_gamma)

    z_data = constants.z_data.values
    fs8_data = constants.fs8_data.values
    errors = 0.5 * (constants.fs8_err_plus + constants.fs8_err_minus)

    z_broadcast = np.array(z_data)[None, :]
    gamma_broadcast = np.array(gamma_vals)[:, None]

    models = growth(z_broadcast, gamma=gamma_broadcast, omega_0=omega0, sigma_8_0=sigma8_0)

    residuals = (models - np.array(fs8_data)[None, :]) / np.array(errors)[None, :]
    chi2_vals = np.sum(residuals**2, axis=1)
    return np.max(chi2_vals)



### Plotting functions

def display(res, title):
    """Displays results as a table. 

    Args:
        res (table): [t1, t2, ...] with ti = [name, value, error]
        title (str): table title
    """
    console = Console()
    table = Table(title=title)

    table.add_column("Variable name", justify="center")
    table.add_column("Value", justify="center")
    table.add_column("Error", justify="center")

    for row in res:
        table.add_row(row[0], f"{row[1]:.3f}", "-" if row[2]=="-" else f"{row[2]:.3f}")
        
    console.print(table)

def display_plot(PDF, ax1_label, ax2_label, ax, xlim=None, ylim=None):
    """_summary_

    Args:
        PDF (_type_): _description_
        x_mean_std (_type_): _description_
        y_mean_std (_type_): _description_
        ax1_label (_type_): _description_
        ax2_label (_type_): _description_
        ax (_type_): _description_
    """
    x_grid = PDF.coords[PDF.dims[1]].values
    y_grid = PDF.coords[PDF.dims[0]].values
    X, Y = np.meshgrid(x_grid, y_grid, indexing='xy')
    Z = PDF.values

    contour = ax.contourf(X, Y, Z, levels=100, cmap='inferno')
    plt.colorbar(contour, ax=ax)

    max_idx = np.unravel_index(np.argmax(Z), Z.shape)
    x_min = X[max_idx]
    y_min = Y[max_idx]

    ax.plot(x_min, y_min, 'ko', label='Best-fit')
    ax.axhline(y_min, color='indigo', linestyle='--')
    ax.axvline(x_min, color='indigo', linestyle='--')

    ax.set_xlabel(ax1_label, fontsize=14)
    ax.set_ylabel(ax2_label, fontsize=14)
    ax.set_title(f"PDF({ax1_label}, {ax2_label})", fontsize=12)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ### Display tables of results

    # display([[ax2_label, y_min, "-"], [ax1_label, x_min, "-"]], "Results for (" + ax1_label +", " + ax2_label +") :")
