<div align="center">
  <img src="https://github.com/VicVEVO/Stage-irap/blob/9d639b217359ae3f725d927e8a783f649d413f8a/images/CosmicWeb.jpg" alt="Cosmic web"  />

# Study of cosmological constraints within the framework of a modified gravity model based on redshift space distortion data.
</div>

# About this project
## Introduction
On sub-horizon scales, in the linear regime, and assuming that dark energy does not cluster, the evolution equation for the growth function is given by:

$$
\frac{d f(a)}{d \ln a} + f^2 + \left( 2 + \frac{1}{2} \frac{d \ln H(a)^2}{d \ln a} \right) f - \frac{3}{2} \Omega_m(a) = 0
$$

Where:
- $\Omega_m(a) \equiv \Omega_{m,0}a^{-3}\frac{H_0^2}{H(a)^2}$
- With:
    - $\Omega_{m,0} \equiv \Omega_m(z=0)$ the **matter density** today.
    - $H(a)$ the Hubble rate as a function of the **scale factor $a$**.

A good approximation for **$f(z)$** is generally:

$$
f(z) \approx \Omega_m^\gamma(z)
$$

In the ΛCDM model, we consider the growth index **$\gamma \approx 0.55$**. The main goal of this project is, by letting $\gamma$ free, see if we still have a **$H_0$** tension (known as **Hubble tension**) in order to consider - or not - a modified gravity model.

## ΛCDM model and the Hubble tension

<img src="https://github.com/VicVEVO/Stage-irap/blob/0f8be1c1d3d29142fa1cea22d6c5db4e0da7b415/images/hubble_tension.png" align="left" width="500em"/>

The **ΛCDM** model (**Lambda Cold Dark Matter**) is the standard cosmological model that extends Einstein’s equations of General Relativity to describe an accelerated expanding universe (Hubble expansion, distribution of galaxies, cosmic microwave background (**CMB**) etc.).

Using the **ΛCDM** model and observations of the early universe, such as the **CMB**, gives one value, while direct measurements based on nearby objects like supernovae and Cepheid stars give a higher value.
<br clear="left"/>

Late Universe Observation: $H_0 = 73.4 \pm 1.0$ km/Mpc/s.

Early Universe Observation: $H_0 = 67.4 \pm 0.5$ km/Mpc/s.

This mismatch might suggest the need for new physics beyond the standard cosmological model.
## Data presentation

<details>
  <summary><strong> 1. Redshift Space Distortion (RSD)</strong></summary>

  We will use a given data collection from [ΛCDM is alive and well](https://arxiv.org/abs/2205.05017) paper, including: 
  - $f\sigma_8$ measured values (`data_rsd['fsig8']`) for given redshifts $z$ (`data_rsd['z']`).
  - Measurement errors $\Delta^+ f\sigma_8$ (`data_rsd['fsig8_err_plus']`) and $\Delta^- f\sigma_8$ (`data_rsd['fsig8_err_minus']`).

  ---
</details>

<details>
  <summary><strong> 2. Type Ia supernova (SN1a)</strong></summary>

  We will use [Pantheon+ DATA](https://github.com/PantheonPlusSH0ES/DataRelease.git), including:
  - A `Pantheon+SH0ES.dat` data file with the given columns:
    - `zCMB`: CMB Corrected Redshift.
    - `m_b_corr`: Tripp1998 corrected/standardized $\mu_b$ magnitude.
    - `CEPH_DIST`: Cepheid calculated absolute distance to host (uncertainty is incorporated in covariance matrix `Pantheon+SH0ES_STAT+SYS.cov`).
    - `IS_CALIBRATOR`: Binary to designate if this **SN** is in a host that has an associated cepheid distance.
  - A `Pantheon+SH0ES_STAT+SYS.cov` file with the covariance matrix stored in.

  ---
</details>

<details>
  <summary><strong> 3. Baryon Acoustic Oscillations (BAO)</strong></summary>

  We will use a given data collection from [DESI 2024 VI: Cosmological Constraints from the Measurements of Baryon Acoustic Oscillations](https://arxiv.org/abs/2404.03002) paper, including: 
  - $\frac{D_m}{r_d}$ measured values (`DM/rd` column) with errors (`DM/rd_err` column) for given redshifts $z$ (`zeff` column).

  ---
</details>

<details>
  <summary><strong> 4. Weak gravitational lensing (SN1a)</strong></summary>

  We will use [DES Y3 + KiDS-1000 fits table](https://des.ncsa.illinois.edu/releases/y3a2/Y3key-joint-des-kids) including: the DES Y3 two-point correlation functions measured from a slightly reduced footprint to remove areal overlap with KiDS-1000.

  This fits file includes:
  - A `xip` (resp. `xim`) column including:
    - $\xi^+$ values (`fits_file['xip']['VALUE']`) (resp. $\xi^-$ with `xim`).
    - Mean $\theta$ values (`fits_file['xip']['ANG']`) and bins (`fits_file['xip']['BIN1']` and `BIN2`) with which $\xi^\pm$ values were measured.
  - A `nz_source_des` column including:
    - $n_{i_{bin}}(z)$ values with $i_{bin} \in \{1, 2, 3, 4\}$ (`fits_file['nz_source_des']['BIN1']` etc.).
    - Mean $z$ values (`fits_file['nz_source_des']['Z_MID']`) with which $n_{i_{bin}}(z)$ were measured.
  - A `nz_source_kids` column including:
    - $n_{i_{bin}}(z)$ values with $i_{bin} \in \{1, 2, 3, 4, 5\}$ (`fits_file['nz_source_kids']['BIN1']` etc.).
    - Mean $z$ values (`fits_file['nz_source_kids']['Z_MID']`) with which $n_{i_{bin}}(z)$ were measured.
  - A `COVMAT` column including $(475,475)$ covariance matrix for the [ξ⁺DES; ξ⁻DES; EₙKiDS] concatenated vector in such order that 200 + 200 + 75 = 475.

  ---
</details>

# Usage

This project is provided as a **Python library** with Jupyter notebooks for demonstrations.

## Requirements

Install dependencies via:

```bash
pip install -r requirements.txt
```

## Core functionality

The utils library is dedicated to computing $\chi^2$ confidence contours in different parameter planes.
We mainly focus on cosmological parameters:

- $\Omega_m$ (`Omega_m`): matter density
- $\sigma_8$ (`sigma_8`): amplitude of density fluctuations
- $\gamma$ (`gamma`): free growth index (to test modified gravity models)

Three grids are supported:

- $(\Omega_m, \sigma_8)$ minimizing over $\gamma$
- $(\sigma_8, \gamma)$ minimizing over $\Omega_m$
- $(\Omega_m, \gamma)$ minimizing over $\sigma_8$

![Basic demo](https://github.com/VicVEVO/Stage-irap/blob/9d639b217359ae3f725d927e8a783f649d413f8a/programs/output/figures/comparison/3-comparison2.png)

## Project tree
The project tree is the following:

    project-root
    |
    ├── papers/             # Some papers
    │
    ├── data/               # Raw data and likelihoods
    │
    ├── output/             # Numerical results and figures
    │   ├── chi2/           # χ² values and grids
    │   └── figures/        # Plots and visualizations
    │
    ├── src/                # Source code
    │   ├── main.ipynb      # Main notebook: how to use the library
    │   ├── experiments.ipynb # Exploratory / test notebooks
    │   └── utils/          # Python modules
    │       ├── __init__.py
    │       ├── cosmo.py
    │       ├── chi2_functions.py
    │       ├── constants.py
    │       └── ...
    │
    └── README.md

All results produced by the project (numerical results and plots) are saved in the `output` directory.

## Module Descriptions (`src/utils/`)

Each Python module in `utils` is independent and modifiable.
Only functions without a leading `_` are considered public.

<details>
  <summary><strong> cosmo.py </strong></summary>
  
  `cosmo.py` Provides cosmological functions (distances, Hubble parameter, growth functions, etc.).
    Forms the physics backbone of the χ² calculation:
  
  You may find further information in the [Cosmological Quantities section](#cosmological-quantities) below.

  ---
</details>

<details>
  <summary><strong> chi2_functions.py </strong></summary>
  
  `chi2_functions.py` implements the Chi2Calculator class.
    Handles $\chi^2$ minimization, grid construction, saving and loading results:

  - `chi2_rsd` function, giving: $\chi^2_{\rm RSD} = \sum_{i} \frac{f\sigma_8(z_i^{\rm obs}) - f\sigma_8^{\rm obs}}{\sigma_i^2}$.
    - With: $\sigma_i^2 = \frac{1}{2} \bigl( \sigma_{f\sigma_8}^{\rm +}(z_i) + \sigma_{f\sigma_8}^{\rm -}(z_i) \bigr)$.
    - $f\sigma_8^{\rm obs}$ = `data_rsd['fsig8']`.
    - $\sigma_{f\sigma_8}^{\rm +}$ = `data_rsd['fsig8_err_plus']` and $\sigma_{f\sigma_8}^{\rm -}$ = `data_rsd['fsig8_err_minus']`.

  - `chi2_panth` function, giving: $\chi^2_{\rm Pantheon+} = (\mu(z^{\rm obs})- \mu^{\rm obs}) ^T C^{-1} (\mu(z^{\rm obs})- \mu^{\rm obs})$.

  - `chi2_bao_dmrd` function, giving: $\chi^2_{\rm DESIDR2} =\sum_{i} \frac{\frac{D_m}{r_d}(z_i^{\rm obs}) - \frac{D_m}{r_d}^{\rm obs}}{\sigma_i^2}$.

  - `compute_chi2_grid_desy3` function.

  - `compute_chi2` function, giving: $\chi^2 = \chi^2_{\rm RSD} + \chi^2_{\rm Pantheon+} + \chi^2_{\rm DESIDR2} + \chi^2_{\rm DESY3 + KiDS-1000}$.

  - `min_chi2_free_gamma` (resp. `min_chi2_free_sigma_8` and `min_chi2_free_Omega_m_0`) function, giving: $\displaystyle \min_{\gamma} \chi^2$ (resp. $\displaystyle \min_{\sigma_8} \chi^2$ and $\displaystyle \min_{\Omega_m} \chi^2$).

  - `display_minimizer` procedure, displaying estimated parameters for a minimized function with [Minuit](https://github.com/scikit-hep/iminuit).

  - `get_minimizer` function, giving the minimized $\chi^2$ value with iminuit depending on which parameters are fixed in $\chi^2$.
  ---
</details>

<details>
  <summary><strong> constants.py </strong></summary>
  
  `constants.py` contains cosmological constants, grid resolution and which parameters are considered as constants when minimizing.

  - `highres` (resp. `lowres`) stands for how broad are the values taken for `Omega_m`, `sigma_8` and `gamma`.

    High quality (`highres`) confidence contours:

    - $\Omega_m \in 0.25, 0.5$
    - $\sigma_8 \in 0.5, 1.05$
    - $\gamma \in 0, 1.2$
   
    Low quality (`lowres`) confidence contours:

    - $\Omega_m \in 0.50, 1$
    - $\sigma_8 \in 0.4, 1.2$
    - $\gamma \in -0.5, 2$
   
  - `is_X_free_minim` means whether X is considered free or fixed when minimizing $\chi^2$.
    
  ---
</details>

<details>
  <summary><strong> data_loader.py </strong></summary>
  
  `data_loader.py` reads and loads raw data in the `data` directory.

  ---
</details>

<details>
  <summary><strong> tools.py </strong></summary>
  
  `tools.py` implements general python functions and procedures:

  - `integral_trapezoid(func, a, b, N, **kwargs)` returns the integral of a given function between two points with the trapezoid approximation.
  - `find_index(x, x_array, delta_x)` returns the corresponding index for an element in a linspace-sorted array.

  ---
</details>

<details>
  <summary><strong> bessel.py </strong></summary>
  
  `bessel.py` implements a `@njit` compatible bessel functions calculator, giving:
  - $J_0(x)$
  
    This function is a wrapper for the [Cephes Mathematical Functions Library](http://www.netlib.org/cephes/) routine `j0.c`.
    The domain is divided into the intervals [0, 5] and [5, $+\infty$]. In the first interval the following rational approximation is used: $J_0(x) \approx (w - r_1^2)(w - r_2^2) \frac{P_3(w)}{Q_8(w)}.$
  
    where:
      - $w = x^2$ and $r_1$, $r_2$ are the roots of $J_0$.
      - $P_3$ (resp. $Q_8$) is a polynom of degree 3 (resp. 8).
   
    In the second interval, the [Hankel asymptotic expansion](https://dlmf.nist.gov/10.17) is employed with two rational functions of degree 6/6 and 7/7.

  - $J(x, n)$
  
  ---
</details>

<details>
  <summary><strong> __init__.py </strong></summary>
  
  `__init__.py` exposes main functions and classes when importing utils.

  You might find all the informations you need in [this example jupyter notebook file](https://github.com/VicVEVO/Stage-irap/blob/b63edcddd36a137bd7af26eac23597a50160b6e5/programs/src/notebooks/main.ipynb).

  ---
</details>

## Configuration
The `cosmo` module and the `Chi2Calculator` class are configurable.
Users can select the datasets to include (`bao`, `rsd`, `pantheon`, `desy3`) and the grid resolution (`N`, `is_highres`).
Example to initialize the $\chi^2$ calculator:

    chi2_calculator = Chi2Calculator(N=20, is_highres=True, rsd=True, pantheon=True)

# Cosmological Quantities
The module `cosmo.py` implements core cosmological quantities required for the χ² analysis.  
Below are the main observables and their definitions:

<details>
  <summary><strong> 1. RSD observable: Growth rate observable fσ₈(z)</strong></summary>
  
  The product of the linear growth rate $f(z)$ and the amplitude of matter fluctuations $\sigma_8(z)$:

  <div align="center">
    $f\sigma_8(z) = f(z) \cdot \sigma_8(z)$
  </div>
  
  where
  
  - $f(z) = \dfrac{d \ln D(z)}{d \ln a}$ is the **growth rate of structure**  
  - $D(z)$ is the linear growth factor normalized at $z=0$  
  - $\sigma_8(z) = D(z) \cdot \sigma_{8,0}$  

  ---
</details>

<details>
  <summary><strong> 2. SN1a observable: Distance modulus μ(z)</strong></summary>

  The distance modulus used in supernova cosmology is defined as:

  <div align="center">
    $\mu(z) = 5 \cdot \log_{10} \left( \frac{d_L(z)}{1 \text{Mpc}} \right) + 25$
  </div>
  
  where $d_L(z)$ is the **luminosity distance**:
  
  <div align="center">
    $d_L(z) = (1+z) \cdot D_M(z)$
  </div>
  
  with $D_M(z)$ the transverse comoving distance.
  
  ---
</details>

<details>
  <summary><strong> 3. BAO observable: 𝐷ₘ/𝑟𝑑</strong></summary>
  
  The comoving angular diameter distance scaled by the sound horizon at the drag epoch $r_d$:
  
  <div align="center">
    $\frac{D_M(z)}{r_d}$
  </div>
  
  In a flat universe ($\Omega_k = 0$), the transverse comoving distance is simply the line-of-sight comoving distance:
  
  <div align="center">
    $D_M(z) = D_C(z) = c \int_0^z \frac{dz'}{H(z')}$
  </div>
  
  The BAO scale also involves the sound horizon at the drag epoch $r_d$:
  
  <div align="center">
    $r_d = \int_{z_d}^{\infty} \frac{c_s(z)}{H(z)} \, dz$
  </div>
  
  with the sound speed in the photon-baryon fluid given by:
  
  <div align="center">
    $c_s(z) = \frac{c}{\sqrt{3 \, \left[1 + R(z)\right]}}$
  </div>
  
  and
  
  <div align="center">
    $R(z) = \frac{3 \rho_b(z)}{4 \rho_\gamma(z)} \;,$
  </div>
  
  where $\rho_b$ is the baryon density and $\rho_\gamma$ the photon density.
  
  ---
</details>

<details>
  <summary><strong> 4. Weak lensing observables: ξ₊ⁱʲ(θ) and ξ₋ⁱʲ(θ)</strong></summary>
  
  Weak gravitational lensing is described through the two-point correlation functions of the shear field,  
  measured between redshift bins $i$ and $j$. They are defined as:
  
  <div align="center">
    $\xi_\pm^{i,j}(\theta) = \sum_{\ell=2}^{\infty} \frac{2\ell+1}{4\pi} \cdot (C_{\ell}^{\epsilon\epsilon}(i,j) \pm C_{\ell}^{\beta\beta}(i,j)) \cdot d^{\ell}_{2 \pm 2}(\theta) $
  </div>
  
  where:
  
  - $\theta$ is the angular separation,
  - $C_{\ell}^{\epsilon\epsilon}(i,j)$ and $C_{\ell}^{\beta\beta}(i,j)$ are the E- and B-mode shear power spectra,  
  - $d^{\ell}_{mn}$ is the reduced Wigner D-matrix,
  - the sum starts at $\ell = 2$ since lensing involves spin-2 fields.

  We can use the **Flat sky approximation** since in both DESY3 and KiDS data, the angular scales analyzed reach small values in the range of a few arcminutes:

  - We replace the expansion in spherical harmonics by an expansion in Fourier modes, giving:
    - $C_{\ell}^{\epsilon\epsilon} \approx \frac{\ell^4}{4} C_{\ell}^{\phi\phi}$
  - The reduced D-matrices for high multipoles can be approximated by Bessel functions:
    - $d^{\ell}_{2,2}(\theta) \approx J_0(\ell\theta)$
    - $d^{\ell}_{2,-2}(\theta) \approx J_4(\ell\theta)$
  
  ---
</details>


These quantities are then combined in `Chi2Calculator` to compare with observational data (BAO, RSD, Supernovae, etc.).

# Acknowledgements

[...]
