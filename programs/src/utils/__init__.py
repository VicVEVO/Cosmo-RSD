from .constants import *
from .data_loader import *
from . import cosmo, chi2_functions, bessel

import numpy as np
from joblib import Parallel, delayed
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from functools import partial
from pathlib import Path

project_root = Path(__file__).parents[2] if "__file__" in globals() else Path().resolve().parents[2]

class Chi2Calculator:
    def __init__(self, N:int, is_highres:bool=False, rsd:bool=False, bao:bool=False, pantheon:bool=False, desy3:bool=False):
        self.N = N
        self.use_rsd = rsd
        self.use_bao = bao
        self.use_pantheon = pantheon
        self.use_desy3 = desy3
        self.chi2_grid = GridConfig(N=N, is_highres=is_highres)
        self.desy3_grid = chi2_functions.compute_chi2_grid_desy3(Om_data_arico, sigma8_data_arico,
                self.chi2_grid.Omega_m_0_vals, self.chi2_grid.sigma_8_vals) ### Temporary
        self.chi2_func = partial(
            chi2_functions.compute_chi2,
            rsd,
            bao,
            pantheon,
            desy3,
            self.chi2_grid.Omega_m_0_vals,
            self.chi2_grid.delta_Omega_m_0,
            self.chi2_grid.sigma_8_vals,
            self.chi2_grid.delta_sigma_8,
            z_data_rsd,
            z_data_panth,
            z_data_bao,
            fs8_data,
            fs8_err_plus,
            fs8_err_minus,
            dmrd_data,
            dmrd_err,
            n_panth,
            is_calibrator_panth,
            m_b_corr_panth,
            ceph_dist_panth,
            inv_cov_panth,
            C,
            chi2_functions.compute_chi2_grid_desy3(Om_data_arico, sigma8_data_arico,
                self.chi2_grid.Omega_m_0_vals, self.chi2_grid.sigma_8_vals)
        )

    def __str__(self):
        return f"<Chi2Calculator use_rsd={self.use_rsd}, use_bao={self.use_bao}, use_pantheon={self.use_pantheon}, use_desy3={self.use_desy3}\n chi2_grid={self.chi2_grid}>"

    def __call__(self, id_grid:int):
        assert id_grid>0 and id_grid<4, ValueError(
            f"Expected grid id between 1 and 3, got id={id_grid}.")
        
        if id_grid == 1:
            params_used = self.get_params_used([True, True, False, is_M_free_minim, is_H0_free_minim, is_rd_free_minim])
            chi2_grid1 = Parallel(n_jobs=-1)(delayed(chi2_functions.min_chi2_free_gamma)(self.chi2_func, Omega_m_0, sigma_8, params_used) for sigma_8 in self.chi2_grid.sigma_8_vals for Omega_m_0 in self.chi2_grid.Omega_m_0_vals)
            self.chi2_grid.chi2_grid1 = np.array(chi2_grid1).reshape(self.chi2_grid.n_s8, self.chi2_grid.n_om)
        elif id_grid == 2:
            params_used = self.get_params_used([True, False, True, is_M_free_minim, is_H0_free_minim, is_rd_free_minim])
            chi2_grid2 = Parallel(n_jobs=-1)(delayed(chi2_functions.min_chi2_free_sigma_8)(self.chi2_func, Omega_m_0, gamma, params_used) for Omega_m_0 in self.chi2_grid.Omega_m_0_vals for gamma in self.chi2_grid.gamma_vals)
            self.chi2_grid.chi2_grid2 = np.array(chi2_grid2).reshape(self.chi2_grid.n_om, self.chi2_grid.n_gamma)
        else:
            params_used = self.get_params_used([False, True, True, is_M_free_minim, is_H0_free_minim, is_rd_free_minim])
            chi2_grid3 = Parallel(n_jobs=-1)(delayed(chi2_functions.min_chi2_free_Omega_m_0)(self.chi2_func, sigma_8, gamma, params_used) for sigma_8 in self.chi2_grid.sigma_8_vals for gamma in self.chi2_grid.gamma_vals)
            self.chi2_grid.chi2_grid3 = np.array(chi2_grid3).reshape(self.chi2_grid.n_s8, self.chi2_grid.n_gamma)
    
    def get_params_used(self, is_used_list:bool):
        assert len(is_used_list)==6, ValueError(
                f"Expected a 6 booleans for the used parameters, got {len(is_used_list)}.")
        params_used = []
        return [["Omega_m_0", is_used_list[0], (self.chi2_grid.om_min, self.chi2_grid.om_max)],
                ["sigma_8", is_used_list[1], (self.chi2_grid.s8_min, self.chi2_grid.s8_max)],
                ["gamma", is_used_list[2], (self.chi2_grid.gamma_min, self.chi2_grid.gamma_max)],
                ["M", is_used_list[3], (self.chi2_grid.M_min, self.chi2_grid.M_max)],
                ["H0", is_used_list[4], (self.chi2_grid.H0_min, self.chi2_grid.H0_max)],
                ["rd", is_used_list[5], (self.chi2_grid.rd_min, self.chi2_grid.rd_max)]]

    def filename(self):
        components = [
            ("B", self.use_bao),
            ("P", self.use_pantheon),
            ("D", self.use_desy3),
            ("R", self.use_rsd),
        ]
        active_components = [letter for letter, flag in components if flag]
        return f"{len(active_components)}-{''.join(active_components)}"

    def set_grid(self, N:int, is_highres:bool):
        """Change grid resolution

        Args:
            N (int): Number of points of numpy linspaces for Omega_m values, etc.
            is_highres (bool): Do we use high resolution parameters range ?
        """
        self.chi2_grid = GridConfig(N=N, is_highres=is_highres)
        self.chi2_func = partial(
            chi2_functions.compute_chi2,
            rsd,
            bao,
            pantheon,
            desy3,
            self.chi2_grid.Omega_m_0_vals,
            self.chi2_grid.delta_Omega_m_0,
            self.chi2_grid.sigma_8_vals,
            self.chi2_grid.delta_sigma_8,
            z_data_rsd,
            z_data_panth,
            z_data_bao,
            fs8_data,
            fs8_err_plus,
            fs8_err_minus,
            dmrd_data,
            dmrd_err,
            n_panth,
            is_calibrator_panth,
            m_b_corr_panth,
            ceph_dist_panth,
            inv_cov_panth,
            C,
            chi2_functions.compute_chi2_grid_desy3(Om_data_arico, sigma8_data_arico,
                self.chi2_grid.Omega_m_0_vals, self.chi2_grid.sigma_8_vals)
        )

    def get_grid(self, id_grid:int):
        assert id_grid>0 and id_grid<4, ValueError(
            f"Expected grid id between 1 and 3, got id={id}.")
        if id_grid == 1:
            return self.chi2_grid.chi2_grid1
        elif id_grid == 2:
            return self.chi2_grid.chi2_grid2
        else:
            return self.chi2_grid.chi2_grid3

    def save_grid(self, id_grid:int, folder=project_root):
        """Save a grid in a given folder.
            Folder format examples:
                PROJ_ROOT/output/chi2/2-BR/lowres/chi2_grid1-(N=50).npz
                PROJ_ROOT/output/chi2/1-P/highres/chi2_grid3-(N=100).npz

        Parameters:
            id_grid (int): The index of the grid
            folder (string)
        """
        output_dir = folder / "output/chi2" / self.filename() / self.chi2_grid.FOLDER
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez(output_dir / f"chi2_grid{str(id_grid)}-(N={str(self.N)}).npz", grid1=self.get_grid(id_grid=id_grid))

    def plot_grid(self, id_grid:int, title:str=r"$\chi^2$ Confidence contours", display_best_chi2:bool=True, savefig:bool=False, folder=project_root, xlim=None, ylim=None, ax=None):
        """Plot chi2 confidence contours according to its given grid.

        Parameters:
            id_grid (int): the grid number
            title (regexp, optional). Defaults to "chi2 Confidence contours".
            display_best_chi2 (bool, optional): Display the coordinates of the minimum chi2. Defaults to True.
            savefig (bool): Do we save the plot as a PNG ? Defaults to False.
            folder: Folder root where the plot is saved. Defaults to the project root.
            xlim ([int, int], optional): The limits in between x must be in order to zoom in/out. Defaults to None.
            ylim ([int, int], optional): The limits in between y must be in order to zoom in/out. Defaults to None.
            ax (axis, optional): Equals to ax[i] with i the subplot index. Defaults to None when used with a single plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        if id_grid == 1:
            chi2_grid = self.chi2_grid.chi2_grid1
            x_vals = self.chi2_grid.Omega_m_0_vals
            y_vals = self.chi2_grid.sigma_8_vals
            labels = [r"$\Omega_m$", r"$\sigma_8$"]
        elif id_grid == 2:
            chi2_grid = self.chi2_grid.chi2_grid2
            x_vals = self.chi2_grid.gamma_vals
            y_vals = self.chi2_grid.Omega_m_0_vals
            labels = [r"$\gamma$", r"$\Omega_m$"]
        else:
            chi2_grid = self.chi2_grid.chi2_grid3
            x_vals = self.chi2_grid.gamma_vals
            y_vals = self.chi2_grid.sigma_8_vals
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

        if savefig:
            fig.savefig(folder / "output/figures" / (self.filename() + "-grid" + str(id_grid) + "-[N=" + str(self.N) + "].png"), bbox_inches='tight')

    def plot_grids(self, title:str=r"$\chi^2$ Confidence contours", display_best_chi2:bool=True, savefig:bool=False, folder=project_root, xlim=None, ylim=None):
        fig, ax = plt.subplots(1, 3, figsize=(10, 5))
        fig.suptitle(title)
        for i in range(1,4):
            self.plot_grid(i, title="", display_best_chi2=display_best_chi2, folder=folder, xlim=xlim, ylim=ylim, ax=ax[i-1])

        if savefig:
            fig.savefig(folder / "output/figures" / (self.filename() + "[N="+ str(self.N) + "]_allplots"+ ".png"), bbox_inches='tight')

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
        
        self.Omega_m_0_vals = np.asarray(np.linspace(self.om_min, self.om_max, self.n_om))
        self.sigma_8_vals = np.asarray(np.linspace(self.s8_min, self.s8_max, self.n_s8))
        self.gamma_vals = np.asarray(np.linspace(self.gamma_min, self.gamma_max, self.n_gamma))
        self.rd_vals = np.asarray(np.linspace(self.rd_min, self.rd_max, self.n_rd))
        self.H0_vals = np.asarray(np.linspace(self.H0_min, self.H0_max, self.n_H0))

        self.delta_Omega_m_0 = self.Omega_m_0_vals[1] - self.Omega_m_0_vals[0]
        self.delta_sigma_8 = self.sigma_8_vals[1] - self.sigma_8_vals[0]
            
    def __str__(self):
        return f"<GridConfig N={self.N}, highres={self.is_highres}, folder='{self.FOLDER}'>"
