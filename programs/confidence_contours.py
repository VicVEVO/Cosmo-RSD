import pandas as pd
import numpy as np
import constants
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as cosmo
from scipy.stats import chi2

def D(z):
    return 1 / (1 + z)

def sigma_8(z):
    return constants.SIGMA_8_0 * D(z)

def S8(omega_m):
    return sigma_8 * np.sqrt(omega_m / 0.3)

def omega_m_z(z, omega_m):
    E = np.sqrt(omega_m * (1 + z)**3 + (1 - omega_m))
    return omega_m * (1 + z)**3 / E**2

def main():
    # Reading data
    df = pd.read_csv("data/fsigma8_data.csv")

    z_data = df["z"].values
    fs8_data = df["f_sigma8"].values
    fs8_err = df["error"].values

    # χ² grid calculation
    omega_m_list = np.linspace(0.1, 0.5, constants.NB_POINTS)
    sigma8_list = np.linspace(0.6, 1.5, constants.NB_POINTS)

    omega_m_grid, sigma8_grid = np.meshgrid(omega_m_list, sigma8_list)
    chi2_grid = np.zeros_like(omega_m_grid)

    # Calcul du χ² sur la grille
    for i in range(constants.NB_POINTS):
        for j in range(constants.NB_POINTS):
            omega_m = omega_m_grid[i, j]
            sigma_8 = sigma8_grid[i, j]
            model = omega_m_z(z_data, omega_m)**constants.GAMMA * sigma_8 * D(z_data)
            chi2_val = np.sum(((fs8_data - model) / fs8_err)**2)
            chi2_grid[i, j] = chi2_val

    S8_grid = sigma8_grid * np.sqrt(omega_m_grid / 0.3)
    delta_chi2 = chi2_grid - np.min(chi2_grid)

    # Niveaux de confiance (2 paramètres libres)
    levels = [chi2.ppf(0.683, df=2), chi2.ppf(0.954, df=2)]  # ≈ 2.30 et 6.18

    # Growth rate plot
    plt.figure(figsize=(8, 5))
    plt.errorbar(z_data, fs8_data, yerr=fs8_err, fmt='o', color='black',
                 label='Données RSD (Tableau 1)')
    plt.xlabel("Redshift $z$")
    plt.ylabel(r"$f\sigma_8(z)$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figure1.png", dpi=300)

    # Confidence Contour plot
    plt.figure(figsize=(7, 5))
    contour = plt.contour(omega_m_grid, S8_grid, delta_chi2, levels=levels,
                          colors=["blue", "red"], linewidths=1.5)
    fmt = {}
    for l, conf in zip(contour.levels, [68, 95]):
        fmt[l] = f"{conf}%"
    plt.clabel(contour, inline=1, fontsize=10, fmt=fmt)

    plt.xlabel(r"$\Omega_m$")
    plt.ylabel(r"$S_8$")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figure2.png", dpi=300)



if __name__ == "__main__":
    main()