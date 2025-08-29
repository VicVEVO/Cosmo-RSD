import numpy as np

from scipy.integrate import quad

from multiprocessing import Pool, cpu_count

from numba import njit

from .tools import integral_trapezoid

### RSD theoretical calculation functions

@njit
def _Omega_m(z, Omega_m_0):
    return Omega_m_0 * (1 + z)**3 / (Omega_m_0 * (1 + z)**3 + 1 - Omega_m_0)

@njit
def _g(z, Omega_m_0):
    omz = _Omega_m(z, Omega_m_0)
    olz = 1 - omz
    return 2.5 * omz / (omz**(4/7) - olz + (1 + omz/2)*(1 + olz/70))

@njit
def _D(z, Omega_m_0):
    return _g(z, Omega_m_0) / _g(0, Omega_m_0) / (1 + np.asarray(z))

@njit
def _sigma8(z, Omega_m_0, sigma_8_0):
    return sigma_8_0 * _D(z, Omega_m_0)

@njit
def _f(z, Omega_m_0, gamma):
    Omega_m_z = _Omega_m(z, Omega_m_0)
    return Omega_m_z ** gamma

@njit
def fs8(z, gamma, Omega_m_0, sigma_8_0):
    """Returns linear growth rate according to the redshift.

    Args:
        z (float): redshift
        gamma (float): γ
        Omega_m_0 (float): Ωm(0)
        sigma_8_0 (float): σ8(0)

    Returns:
        float: fσ8(z)
    """
    return _f(z, Omega_m_0, gamma) * _sigma8(z, Omega_m_0, sigma_8_0)

### SN1A theoretical calculation functions

@njit
def _E_LCDM(z, Omega_m_0):
    return np.sqrt(Omega_m_0 * (1.0 + z)**3 + (1.0 - Omega_m_0))

@njit
def integral_trapezoid_inv_H_LCDM(a, b, N, c, H0, Omega_m_0):
    step = (b - a) / N
    result = 0.5 * (1.0 / _E_LCDM(a, Omega_m_0) + 1.0 / _E_LCDM(b, Omega_m_0))
    for i_step in range(1, N):
        result += 1.0 / _E_LCDM(a + i_step * step, Omega_m_0)
    return (c / H0) * step * result

@njit
def _H_LCDM(z, H0, Omega_m_0):
    return H0 * np.sqrt(Omega_m_0 * (1+z)**3 + (1 - Omega_m_0))

@njit
def _inv_H_LCDM(z, H0, Omega_m_0):
    return 1.0 / _H_LCDM(z, H0, Omega_m_0)

@njit
def _dL(z, H0, Omega_m_0, c):
    integral = integral_trapezoid_inv_H_LCDM(0.0, z, 100, c=c, H0=H0, Omega_m_0=Omega_m_0)
    return (1 + z) * integral

@njit
def mu(z, Omega_m_0, H0, c):
    return 5 * np.log10(_dL(z, H0, Omega_m_0, c)) + 25

### BAO theoretical calculation functions

@njit
def _Dmrd(z, Omega_m_0, rd, H0, c):
    integral = integral_trapezoid_inv_H_LCDM(0.0, z, 100, c=c, H0=H0, Omega_m_0=Omega_m_0)
    return integral / rd

@njit
def Dmrd_array(z_array, Omega_m_0, rd, H0, c):
    result = np.empty_like(z_array)
    for i in range(z_array.size):
        result[i] = _Dmrd(z_array[i], Omega_m_0, rd, H0, c)
    return result

### Weak Lensing theoretical calculation functions

@njit
def _z_d(Omega_m_0, Omega_b_0, h):
    """Drag epoch redshift

    Args:
        Omega_m_0 (float)
        Omega_b_0 (float)
        h (float)

    Returns:
        float: z_d
    """
    omega_m_0 = Omega_m_0 * h**2
    omega_b_0 = Omega_b_0 * h**2

    b1 = 0.313 * omega_m_0**(-0.419) * (1 + 0.607 * omega_m_0**0.674)
    b2 = 0.238 * omega_m_0**0.223

    return (1291 * omega_m_0**0.251) / (1 + 0.659 * omega_m_0**0.828) * (1 + b1 * omega_b_0**b2)

@njit
def _s(omega_m_0, f_baryon):
    """Sound scale approximation (Mpc)

    Args:
        omega_m_0 (float)
        f_baryon (float): omega_b_0/omega_m_0

    Returns:
        float: s
    """
    return 44.5 * np.log(9.83/omega_m_0) / np.sqrt(1 + 10.0 * (omega_m_0*f_baryon)**0.75)

@njit
def _alpha_gamma(omega_m_0, omega_b_0):
    """α_Γ coefficient (Eisenstein & Hu 1998 Eq. 31)

    Args:
        omega_m_0 (float)
        omega_b_0 (float)

    Returns:
        float: α_Γ
    """
    return 1 - 0.328 * np.log(431 * omega_m_0) * (omega_b_0 / omega_m_0) + 0.38 * np.log(22.3 * omega_m_0) * (omega_b_0 / omega_m_0)**2

@njit
def _T_k(k, Omega_m_0, Omega_b_0, Omega_gamma, h):
    """T(k)

    Args:
        k (float)
        Omega_m_0 (float)
        Omega_b_0 (float)
        Omega_gamma (float)
        h (float)

    Returns:
        float: T(k)
    """
    omega_m_0 = Omega_m_0 * h**2
    f_baryon = Omega_b_0/Omega_m_0
    k_equality = 0.0746 * omega_m_0
    q = k / 13.41 / k_equality
    alpha_gamma_val = _alpha_gamma(omega_m_0, Omega_b_0*h**2)
    q_eff = q / (alpha_gamma_val + (1-alpha_gamma_val) / (1 + (0.43*k * _s(omega_m_0, f_baryon))**4))
    T_nowiggles_L0 = np.log(2.0*np.e + 1.8*q_eff)
    T_nowiggles_C0 = 14.2 + 731.0 / (1 + 62.5 * q_eff)
    return T_nowiggles_L0 / (T_nowiggles_L0 + T_nowiggles_C0 * q_eff**2)

@njit
def _P_k_z(k, A_s, k_star, n_s, Tdm):
    """P(k, z)

    Args:
        k (float)
        A_s (float)
        k_star (float)
        n_s (float)
        Tdm (float)

    Returns:
        float: P(k, z)
    """
    P_R = (2*np.pi**2 / k**3) * A_s * (k/k_star)**(n_s - 1)
    return P_R * Tdm**2

@njit
def _T_k_z(k, z, Omega_m_0, Omega_b_0, Omega_gamma, h, c, fact):
    """T(k, z)

    Args:
        k (float)
        z (float)
        Omega_m_0 (float)
        Omega_b_0 (float)
        Omega_gamma (float)
        h (float)
        c (float)
        fact (float): _description_

    Returns:
        float: T(k, z)
    """
    return fact * 2/5 * k**2 / ((100 * h / c)**2 * Omega_m_0) * _T_k(k, Omega_m_0, Omega_b_0, Omega_gamma, h) * _D(z, Omega_m_0)

@njit
def P(k, z, Omega_m_0, Omega_b_0, Omega_gamma, h, c, n_s, k_star, A_s, fact):
    T = _T_k_z(k=k, z=z, Omega_m_0=Omega_m_0, Omega_b_0=Omega_b_0, Omega_gamma=Omega_gamma, h=h, c=c, fact=fact)
    return _P_k_z(k=k, A_s=A_s, k_star=k_star, n_s=n_s, Tdm=T)