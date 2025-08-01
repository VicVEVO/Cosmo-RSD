from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
from iminuit import Minuit


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


def save_grid(grid, number, folder):
     """Save a grid in a given folder.

     Parameters:
         grid (array)
         number (int): The index of the grid
         folder (string)
     """
     np.savez(folder / f"chi2_grid{str(number)}.npz", grid1=chi2_grid1)


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