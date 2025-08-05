import pandas as pd
from pathlib import Path
import numpy as np
import h5py

# ======================================================================
# ROOT SETUP
# ======================================================================
project_root = Path(__file__).parents[2] if "__file__" in globals() else Path().resolve().parents[2]
data_root = project_root / "data"

# ======================================================================
# RSD DATA
# ======================================================================
data_rsd = pd.read_csv(data_root / "fsigma8_data.dat", sep=';', header=0)

z_data_rsd = np.asarray(data_rsd['z'])
fs8_data = np.asarray(data_rsd['fsig8'])
fs8_err_plus = np.asarray(data_rsd['fsig8_err_plus'])
fs8_err_minus = np.asarray(data_rsd['fsig8_err_minus'])

# ======================================================================
# Pantheon+ DATA
# ======================================================================
n_panth = 1701

cov_mat = pd.read_csv(data_root / "DataRelease/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES_STAT+SYS.cov")
cov_mat = np.array(cov_mat).reshape(n_panth, n_panth)
inv_cov_panth = np.asarray(np.linalg.inv(cov_mat))

data_pantheon = pd.read_csv(data_root / "DataRelease/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat", sep=' ', header=0)

z_data_panth = np.asarray(data_pantheon['zCMB'])
m_b_corr_panth = np.asarray(data_pantheon['m_b_corr'])
mu_data_panth = np.asarray(data_pantheon['MU_SH0ES'])
ceph_dist_panth = np.asarray(data_pantheon['CEPH_DIST'])
is_calibrator_panth = np.asarray(data_pantheon['IS_CALIBRATOR'])

# ======================================================================
# DESIDR2 DATA
# ======================================================================
data_bao = pd.read_csv(data_root / "DESIDR2/alpha_data.dat", sep=';', header=0)

z_data_bao = np.asarray(data_bao['zeff'])
Dmrd_data = np.asarray(data_bao['DM/rd'])
Dmrd_err = np.asarray(data_bao['DM/rd_err'])

# ======================================================================
# DESY3 arico DATA
# ======================================================================
with h5py.File(data_root / 'posteriors_DESY3_arico.hdf5', 'r') as f:
     S8_data_arico = f['S8'][:]
     Om_data_arico = f['omega_matter'][:]
     sigma8_data_arico = S8_data_arico * np.sqrt(.3/Om_data_arico)