from . import cosmo
from . import constants
from . import tools

from numba import njit
import numpy as np
from iminuit import Minuit
from scipy.stats import gaussian_kde

# ======================================================================
# CHI2 CALCULATION FUNCTIONS
# ======================================================================
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
    model = cosmo.fs8(z_data, gamma=gamma, omegam_0=omega_m, sigma8_0=sigma_8)

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
            delta_mu[i] = cosmo.mu(z_data_panth[i], omega_m, H0, c) - (m_b_corr_panth[i] - M)
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
    model = cosmo.Dmrd_array(z_data, omega_m, rd, H0, c)
    residuals = (model - dmrd_data) / dmrd_err
    return np.sum(residuals**2)

def compute_chi2_grid_desy3(omega_m_data, sigma_8_data, omega_m_vals, sigma_8_vals):
    X, Y = np.meshgrid(omega_m_vals, sigma_8_vals)
    positions = np.vstack([X.ravel(), Y.ravel()])

    values = np.vstack([omega_m_data, sigma_8_data])
    kde = gaussian_kde(values)
    density = kde(positions).reshape(X.shape)

    P = density / np.max(density)
    with np.errstate(divide='ignore'):
        delta_chi2_grid_desy3 = -2 * np.log(P)
        delta_chi2_grid_desy3[np.isinf(delta_chi2_grid_desy3)] = np.nanmax(delta_chi2_grid_desy3) + 1
    
    return np.asarray(delta_chi2_grid_desy3)

@njit
def compute_chi2(use_rsd, use_bao, use_panth, use_desy3, omega_m_array, delta_omega_m, sigma_8_array, delta_sigma_8, z_data_rsd, z_data_panth, z_data_bao, fs8_data, fs8_err_plus, fs8_err_minus, dmrd_data, dmrd_err, n_panth, is_calibrator_panth, m_b_corr_panth, ceph_dist_panth, inv_cov_panth, c, chi2_grid_desy3, omega_m, sigma_8, gamma, rd, H0, M):
    chi2_val = 0.0
    if use_rsd:
        chi2_val += chi2_rsd(z_data_rsd, fs8_data, fs8_err_plus, fs8_err_minus, omega_m, sigma_8, gamma)
    if use_panth:
        chi2_val += chi2_panth(n_panth, z_data_panth, is_calibrator_panth, m_b_corr_panth, ceph_dist_panth, inv_cov_panth, omega_m, H0, M, c)
    if use_bao:
        chi2_val += chi2_bao_dmrd(z_data_bao, dmrd_data, dmrd_err, c, omega_m, rd, H0)
    if use_desy3:
        i_omega = tools.find_index(omega_m, omega_m_array, delta_omega_m)
        i_sigma = tools.find_index(sigma_8, sigma_8_array, delta_sigma_8)
        chi2_val += chi2_grid_desy3[i_omega, i_sigma]
    return chi2_val


def chi2_for_const_gamma(chi2_func, omega_m, sigma_8, params_used):
    """Returns minimum chi2 value for a given Omega_m, sigma_8 with a free gamma, rd, H0, M.

    Args:
        chi2_func (fun)
        omega_m (float)
        sigma_8 (float)
        params_used (list): 
            - [L_1, .., L_6]
            with L_i = [str, bool, (float, float)]
                - [parameter name, is it fixed ?, its range if it is free] 

    Returns:
        float: the minimum chi2 value
    """
    initial_params = {"omega_m":omega_m, "sigma_8":sigma_8, "gamma":constants.GAMMA, "rd":constants.RD, "H0":constants.H0, "M":constants.M}
    minimizer = get_minimizer(chi2_func, params_used, initial_params)
    minimizer.migrad()
    return minimizer.fval

def chi2_for_const_sigma_8(chi2_func, omega_m, gamma, params_used):
    """Returns minimum chi2 value for a given Omega_m, gamma with a free sigma_8, rd, H0, M.

    Args:
        chi2_func (fun)
        omega_m (float)
        gamma (float)
        params_used (list): 
            - [L_1, .., L_6]
            with L_i = [str, bool, (float, float)]
                - [parameter name, is it fixed ?, its range if it is free] 

    Returns:
        float: the minimum chi2 value
    """
    initial_params = {"omega_m":omega_m, "sigma_8":constants.SIGMA_8_0, "gamma":gamma, "rd":constants.RD, "H0":constants.H0, "M":constants.M}
    minimizer = get_minimizer(chi2_func, params_used, initial_params)
    minimizer.migrad()
    return minimizer.fval

def chi2_for_const_omega_m(chi2_func, sigma_8, gamma, params_used):
    """Returns minimum chi2 value for a given sigma_8, gamma with a free Omega_m, rd, H0, M.

    Args:
        chi2_func (fun)
        sigma_8 (float)
        gamma (float)
        params_used (list): 
            - [L_1, .., L_6]
            with L_i = [str, bool, (float, float)]
                - [parameter name, is it fixed ?, its range if it is free] 

    Returns:
        float: the minimum chi2 value
    """
    initial_params = {"omega_m":constants.OMEGAM_0, "sigma_8":sigma_8, "gamma":gamma, "rd":constants.RD, "H0":constants.H0, "M":constants.M}
    minimizer = get_minimizer(chi2_func, params_used, initial_params)
    minimizer.migrad()
    return minimizer.fval

# ======================================================================
# TOOLS FOR CHI2 GRIDS
# ======================================================================

def plot_chi2_contours(chi2_grid, x_vals, y_vals, labels, title=r"$\chi^2$ Confidence contours", display_best_chi2=True, xlim=None, ylim=None, ax=None):
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

# ======================================================================
# TOOLS FOR MINIMIZING WITH MINUIT
# ======================================================================

def display_minimizer(minimizer, epsilon=3):
     """Display estimated parameters for a Minuit minimizer.

     Parameters:
          minimizer (Minuit)
          epsilon (int, optional): Parameter precision. Defaults to 3.
     """
     assert isinstance(minimizer, Minuit), ValueError(
          f"Expected a Minuit object, got {type(minimizer).__name__}")
     
     print(f"Fit results (Chi2 = {minimizer.fval:.{epsilon}f}):")
     for param in minimizer.parameters:
          print(f"{param} = {minimizer.values[param]:.{epsilon}f} ± {minimizer.errors[param]:.{epsilon}f}")


def get_minimizer(chi2_func, params_used, initial_params):
    """Minimize chi2_func with iminuit depending on which parameters are fixed.

    Parameters:
        chi2_func (fun): _description_
        params_used: [x1_used, x2_used, ...]
            x_used (array): ["x", bool, (x_min, x_max)].
                If bool==True then x is fixed.
                Otherwise it is free, between x_min and x_max
        initial_params: {"x": float} where the float corresponds to the initial
            value of x for the minimization.

    Returns:
        Minuit minimizer
    """
    assert len(params_used)==6, ValueError(
          f"Expected 6 parameters, got {len(params_used)}.")
    
    minimizer = Minuit(chi2_func, omega_m=initial_params["omega_m"], sigma_8=initial_params["sigma_8"], gamma=initial_params["gamma"], rd=initial_params["rd"], H0=initial_params["H0"], M=initial_params["M"])
    for param_used in params_used:
        if param_used[1]:
            minimizer.fixed[param_used[0]] = True
        else:
            minimizer.limits[param_used[0]] = param_used[2]
    return minimizer
