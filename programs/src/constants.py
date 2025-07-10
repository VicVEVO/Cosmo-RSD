import pandas as pd
import numpy as np

# Planck 2018
OMEGA_0 = 0.315 # Ωₘ
OMEGA_0_SIGMA = .007

SIGMA_8_0 = 0.811 # σ8
SIGMA_8_0_SIGMA = .006

GAMMA = 0.55 # γ

C = 299792.458

### RSD
data_rsd = pd.read_csv("../data/fsigma8_data.dat", sep=';', header=0)

z_data = data_rsd['z']
fs8_data = data_rsd['fsig8']
fs8_err_plus = data_rsd['fsig8_err_plus']
fs8_err_minus = data_rsd['fsig8_err_minus']
fs8_err = [fs8_err_plus, fs8_err_minus]

### Pantheon+
n_panth = 1701

cov_mat = pd.read_csv("../data/DataRelease/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES_STAT+SYS.cov")
cov_mat = np.array(cov_mat).reshape(n_panth, n_panth)
inv_cov_panth = np.linalg.inv(cov_mat)

data_pantheon = pd.read_csv("../data/DataRelease/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat", sep=' ', header=0)

z_data_panth = data_pantheon['zCMB']
m_b_corr_panth = data_pantheon['m_b_corr']
mu_data_panth = data_pantheon['MU_SH0ES']
ceph_dist_panth = data_pantheon['CEPH_DIST']
is_calibrator_panth = data_pantheon['IS_CALIBRATOR']