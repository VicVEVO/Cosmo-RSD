import pandas as pd

# Planck 2018
OMEGA_0 = 0.315 # Ωₘ
OMEGA_0_SIGMA = .007

SIGMA_8_0 = 0.811 # σ8
SIGMA_8_0_SIGMA = .006

GAMMA = 0.55 # γ

C = 299792458

data_rsd = pd.read_csv("../data/fsigma8_data.dat", sep=';', header=0)
data_pantheon = pd.read_csv("../data/DataRelease/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat", sep=' ', header=0)
cov_mat = pd.read_csv("../data/DataRelease/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES_STAT+SYS.cov")

z_data_panth = data_pantheon['zCMB']
mu_data_panth = data_pantheon['MU_SH0ES']


z_data = data_rsd['z']
fs8_data = data_rsd['fsig8']
fs8_err_plus = data_rsd['fsig8_err_plus']
fs8_err_minus = data_rsd['fsig8_err_minus']
fs8_err = [fs8_err_plus, fs8_err_minus]