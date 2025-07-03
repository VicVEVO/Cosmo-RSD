import pandas as pd

# Planck 2018
OMEGA_0 = 0.315 # Ωₘ
OMEGA_0_SIGMA = .007

SIGMA_8_0 = 0.811 # σ8
SIGMA_8_0_SIGMA = .006

GAMMA = 0.55 # γ

data = pd.read_csv("data/fsigma8_data.dat", sep=';', header=0)
z_data = data['z']
fs8_data = data['fsig8']
fs8_err_plus = data['fsig8_err_plus']
fs8_err_minus = data['fsig8_err_minus']
fs8_err = [fs8_err_plus, fs8_err_minus]