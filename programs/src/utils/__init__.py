from .constants import *
from .data_loader import *
from . import cosmo, chi2_functions

import numpy as np
from joblib import Parallel, delayed
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

class GridConfig:
    def __init__(self, N: int, is_highres: bool):
        self.N = N
        self.is_highres = is_highres

        self.n_om = N
        self.n_s8 = N
        self.n_gamma = N
        self.n_H0 = N
        self.n_rd = N

        self.chi2_grid1 = np.empty((self.n_s8, self.n_om))
        self.chi2_grid2 = np.empty((self.n_om, self.n_gamma))
        self.chi2_grid3 = np.empty((self.n_s8, self.n_gamma))

        self.H0_min = H0_min
        self.H0_max = H0_max
        self.rd_min = rd_min
        self.rd_max = rd_max
        self.M_min = M_min
        self.M_max = M_max

        if is_highres:
            self.om_min = om_min_highres
            self.om_max = om_max_highres
            self.s8_min = s8_min_highres
            self.s8_max = s8_max_highres
            self.gamma_min = gamma_min_highres
            self.gamma_max = gamma_max_highres
            self.FOLDER = "highres"
        else:
            self.om_min = om_min_lowres
            self.om_max = om_max_lowres
            self.s8_min = s8_min_lowres
            self.s8_max = s8_max_lowres
            self.gamma_min = gamma_min_lowres
            self.gamma_max = gamma_max_lowres
            self.FOLDER = "lowres"
        
        self.omega_m_vals = np.asarray(np.linspace(self.om_min, self.om_max, self.n_om))
        self.sigma_8_vals = np.asarray(np.linspace(self.s8_min, self.s8_max, self.n_s8))
        self.gamma_vals = np.asarray(np.linspace(self.gamma_min, self.gamma_max, self.n_gamma))
        self.rd_vals = np.asarray(np.linspace(self.rd_min, self.rd_max, self.n_rd))
        self.H0_vals = np.asarray(np.linspace(self.H0_min, self.H0_max, self.n_H0))
            
    def __str__(self):
        return f"<GridConfig N={self.N}, highres={self.is_highres}, folder='{self.FOLDER}'>"

    def get_grid(self, id_grid:int):
        assert id_grid>0 and id_grid<4, ValueError(
            f"Expected grid id between 1 and 3, got id={id}.")
        if id_grid == 1:
            return self.chi2_grid1
        elif id_grid == 2:
            return self.chi2_grid2
        else:
            return self.chi2_grid3

    def get_params_used(self, is_used_list:bool):
        assert len(is_used_list)==6, ValueError(
                f"Expected a 6 booleans for the used parameters, got {len(is_used_list)}.")
        params_used = []
        return [["omega_m", is_used_list[0], (self.om_min, self.om_max)],
                ["sigma_8", is_used_list[1], (self.s8_min, self.s8_max)],
                ["gamma", is_used_list[2], (self.gamma_min, self.gamma_max)],
                ["M", is_used_list[3], (self.M_min, self.M_max)],
                ["H0", is_used_list[4], (self.H0_min, self.H0_max)],
                ["rd", is_used_list[5], (self.rd_min, self.rd_max)]]

    def compute(self, chi2_func, id_grid:int):
        assert id_grid>0 and id_grid<4, ValueError(
            f"Expected grid id between 1 and 3, got id={id}.")

        if id_grid == 1:
            params_used = self.get_params_used([True, True, False, False, False, True])
            chi2_grid1 = Parallel(n_jobs=-1)(delayed(chi2_functions.chi2_for_const_gamma)(chi2_func, omega_m, sigma_8, params_used) for sigma_8 in self.sigma_8_vals for omega_m in self.omega_m_vals)
            self.chi2_grid1 = np.array(chi2_grid1).reshape(self.n_s8, self.n_om)
        elif id_grid == 2:
            params_used = self.get_params_used([True, False, True, False, False, True])
            chi2_grid2 = Parallel(n_jobs=-1)(delayed(chi2_functions.chi2_for_const_sigma_8)(chi2_func, omega_m, gamma, params_used) for omega_m in self.omega_m_vals for gamma in self.gamma_vals)
            self.chi2_grid2 = np.array(chi2_grid2).reshape(self.n_om, self.n_gamma)
        else:
            params_used = self.get_params_used([False, True, True, False, False, True])
            chi2_grid3 = Parallel(n_jobs=-1)(delayed(chi2_functions.chi2_for_const_omega_m)(chi2_func, sigma_8, gamma, params_used) for sigma_8 in self.sigma_8_vals for gamma in self.gamma_vals)
            chi2_grid3 = np.array(chi2_grid3).reshape(self.n_s8, self.n_gamma)

    def save(self, id_grid:int, folder):
        """Save a grid in a given folder.

        Parameters:
            id_grid (int): The index of the grid
            folder (string)
        """
        np.savez(folder / self.FOLDER / f"chi2_grid{str(id_grid)}.npz", grid1=self.get_grid(id_grid=id_grid))

    def plot_contours(self, id_grid, title=r"$\chi^2$ Confidence contours", display_best_chi2=True, xlim=None, ylim=None, ax=None):
        """Plot chi2 confidence contours according to its given grid.

        Parameters:
            chi2_grid (array): 2D numpy arraylike
            x_vals (array): 1D numpy arraylike
            y_vals (array): 1D numpy arraylike
            labels ([str, str]): The labels ["x_label", "y_label"]
            title (regexp, optional). Defaults to "chi2 Confidence contours".
            display_best_chi2 (bool, optional): Display the coordinates of the minimum chi2. Defaults to True.
            xlim ([int, int], optional): The limits in between x must be in order to zoom in/out. Defaults to None.
            ylim ([int, int], optional): The limits in between y must be in order to zoom in/out. Defaults to None.
            ax (axis, optional): Equals to ax[i] with i the subplot index. Defaults to None when used with a single plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        if id_grid == 1:
            chi2_grid = self.chi2_grid1
            x_vals = self.omega_m_vals
            y_vals = self.sigma_8_vals
            labels = [r"$\Omega_m$", r"$\sigma_8$"]
        elif id_grid == 2:
            chi2_grid = self.chi2_grid2
            x_vals = self.gamma_vals
            y_vals = self.omega_m_vals
            labels = [r"$\gamma$", r"$\Omega_m$"]
        else:
            chi2_grid = self.chi2_grid3
            x_vals = self.gamma_vals
            y_vals = self.sigma_8_vals
            labels = [r"$\gamma$", r"$\sigma_8$"]

        chi2_grid -= np.min(chi2_grid)

        levels = [2.3, 6.17, 11.8]
        colors = ['khaki', 'lightsalmon', 'mediumpurple']

        chi2_clipped = np.clip(chi2_grid, a_min=None, a_max=levels[2])

        cf = ax.contourf(x_vals, y_vals, chi2_clipped, levels=100, cmap='inferno_r')

        for level, color in zip(levels, colors):
            ax.contour(x_vals, y_vals, chi2_grid, levels=[level], colors=[color], linewidths=2)

        if display_best_chi2:
            min_idx = np.unravel_index(np.nanargmin(chi2_grid), chi2_grid.shape)
            best_omega = x_vals[min_idx[1]]
            best_sigma = y_vals[min_idx[0]]
            ax.plot(best_omega, best_sigma, 'ko', label='Best-fit')
            ax.axhline(best_sigma, color='indigo', linestyle='--', alpha=0.6)
            ax.axvline(best_omega, color='indigo', linestyle='--', alpha=0.6)

        legend_handles = [
            Patch(color='khaki', label=r'$1\sigma$'),
            Patch(color='lightsalmon', label=r'$2\sigma$'),
            Patch(color='mediumpurple', label=r'$3\sigma$')
        ]
        ax.legend(handles=legend_handles, loc='upper right')

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_title(title)
        ax.set_facecolor('black')

        plt.tight_layout()