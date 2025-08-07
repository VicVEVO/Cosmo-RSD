# ======================================================================
# COSMOLOGICAL CONSTANTS (Planck 2018)
# ======================================================================
OMEGAM_0 = 0.315
OMEGAM_0_ERR = .007

SIGMA_8_0 = 0.811
SIGMA_8_0_ERR = .006

GAMMA = 0.55

RD = 147.4
H0 = 73.4
M = -19.25

C = 299792.458

# ======================================================================
# Chi2 GRID CONSTANTS
# ======================================================================
H0_min, H0_max = 50, 100
rd_min, rd_max = 100, 200
M_min, M_max = -30, -10

om_min_highres, om_max_highres = 0.25, 0.4 # 0.25, 0.5
s8_min_highres, s8_max_highres = 0.6, 0.9  # 0.5, 1.05
gamma_min_highres, gamma_max_highres = 0, 1.2

om_min_lowres, om_max_lowres = 0.05, 1
s8_min_lowres, s8_max_lowres = 0.4, 1.2
gamma_min_lowres, gamma_max_lowres = -0.5, 2

# ======================================================================
# MINIMIZER CONSTANTS
# ======================================================================

is_M_free_minim = False
is_H0_free_minim = False
is_rd_free_minim = True