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

def gamma(omega_m_z):
    omega_m_z = np.asarray(omega_m_z)
    return 6/11 - 15/2057 * np.log(omega_m_z)

def make_constant_gamma_func(gamma_value):
    return lambda omega_m_z: gamma_value

def sigma_8(z, omega_0, sigma_8_0):
    return sigma_8_0 * D(z, omega_0)

def f(gamma_func, z, omega_0):
    omega_m_z = omega_m(z, omega_0)
    return omega_m_z ** gamma_func(omega_m_z)

def growth(z, gamma_func = gamma, omega_0 = constants.OMEGA_0, sigma_8_0 = constants.SIGMA_8_0):
    """Returns linear growth rate according to the redshift.

    Args:
        z (float): redshift
        fun_gamma (γ): gamma. Defaults to gamma.
        omega_0 (float): Ωm(0). Defaults to constants.OMEGA_0.
        sigma_8_0 (float): σ8(0). Defaults to constants.SIGMA_8_0.

    Returns:
        float: fσ8(z)
    """
    return f(gamma_func, z, omega_0) * sigma_8(z, omega_0, sigma_8_0)

### Chi2

def calc_chi2_gamma_sigma(sigma_8, gamma_value):
    gamma_func = make_constant_gamma_func(gamma_value)
    errors = 0.5 * (constants.fs8_err_plus + constants.fs8_err_minus)
    chi2 = np.sum((growth(constants.z_data.values, gamma_func, sigma_8_0 = sigma_8) - constants.fs8_data)**2 / errors**2)
    return chi2

def calc_chi2_gamma_omega(omega_0, gamma_value):
    gamma_func = make_constant_gamma_func(gamma_value)
    errors = 0.5 * (constants.fs8_err_plus + constants.fs8_err_minus)
    chi2 = np.sum((growth(constants.z_data.values, gamma_func, omega_0) - constants.fs8_data)**2 / errors**2)
    return chi2

def calc_chi2_sigma_omega_vectorized(omega0, sigma8_0):
    omega0 = np.asarray(omega0)
    sigma8_0 = np.asarray(sigma8_0)
    errors = 0.5 * (constants.fs8_err_plus + constants.fs8_err_minus)
    
    model = growth(constants.z_data.values, omega_0=omega0[..., None], sigma_8_0=sigma8_0[..., None])
    residuals = (model - constants.fs8_data) / errors
    chi2 = np.sum(residuals**2)
    return chi2

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

def display_plot(PDF, x_mean_std, y_mean_std, ax1_label, ax2_label, ax):
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

    contour = ax.contourf(X, Y, Z, levels=30, cmap='inferno')
    plt.colorbar(contour, ax=ax)


    x_mean, x_std = x_mean_std
    y_mean, y_std = y_mean_std
    ax.plot(x_mean, y_mean, 'ko', label='Best-fit')
    ax.axhline(y_mean, color='indigo', linestyle='--')
    ax.axvline(x_mean, color='indigo', linestyle='--')

    ax.set_xlabel(ax1_label, fontsize=14)
    ax.set_ylabel(ax2_label, fontsize=14)
    ax.set_title(f"PDF({ax1_label}, {ax2_label})", fontsize=12)

