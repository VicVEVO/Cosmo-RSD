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

</details>

<details>
  <summary><strong> 2. Type Ia supernova (SN1a)</strong></summary>

</details>

<details>
  <summary><strong> 3. Baryon Acoustic Oscillations (BAO)</strong></summary>

</details>

<details>
  <summary><strong> 4. Weak gravitational lensing (SN1a)</strong></summary>

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
We focus mainly on cosmological parameters:

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

- `cosmo.py`
    Provides cosmological functions (distances, Hubble parameter, growth functions, etc.).
    Forms the physics backbone of the χ² calculation.

- `chi2_functions.py`
    Implements the Chi2Calculator class.
    Handles $\chi^2$ minimization, grid construction, saving and loading results.

- `constants.py`
    Contains cosmological constants, grid resolution and which parameters are considered as constants when minimizing.

- `data_loader.py`
    Reads and loads raw data in the `data` directory.

- `tools.py`
    Implements general python functions and procedures. 

- `polynoms.py`
    Implements a simplified Python polynom class to provide associated legendre polynoms for weak gravitational lensing data. 

- `__init__.py`
    Exposes main functions and classes when importing utils.

## Configuration
The `cosmo` module and the `Chi2Calculator` class are configurable.
Users can select the datasets to include (`bao`, `rsd`, `pantheon`, `desy3`) and the grid resolution (`N`, `is_highres`).
Example to initialize the $\chi^2$ calculator:

    chi2_calculator = Chi2Calculator(N=20, is_highres=True, rsd=True, pantheon=True)

# Cosmological Quantities in `cosmo.py`
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
    $\xi_\pm^{i,j}(\theta) = \sum_{\ell=2}^{\infty} \frac{2\ell+1}{2\pi \ell^2 (\ell+1)^2} \cdot (G^{+}_{\ell,2}(\cos \theta) \pm G^{-}_{\ell,2}(\cos \theta)) \cdot (C_{EE}^{i,j}(\ell) \pm C_{BB}^{i,j}(\ell))$
  </div>
  
  where:
  
  - $\theta$ is the angular separation,
  - $C_{EE}^{i,j}(\ell)$ and $C_{BB}^{i,j}(\ell)$ are the E- and B-mode shear power spectra,  
  - $G_{\ell,2}^{+}$ and $G_{\ell,2}^{-}$ are **geometrical kernels** related to spin-2 spherical harmonics (`4.19`,  	[arXiv:astro-ph/9609149](https://arxiv.org/abs/astro-ph/9609149))
  - the sum starts at $\ell = 2$ since lensing involves spin-2 fields.
  
  ---
</details>


These quantities are then combined in `Chi2Calculator` to compare with observational data (BAO, RSD, Supernovae, etc.).

# Acknowledgements

[...]
