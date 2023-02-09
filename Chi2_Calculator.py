"""
for T-fits:
    chi^2 = 1.0/(N-p-1) * sum( (x_i-f(x_i))^2/sigma_i^2 )

    with:
    N ... no. of data points
    p ... no. of fitted parameters
    sigma ... errors for lattice points

chi^2 ~ 1 represents a good fit
for statistics see e.g.    http://neutrons.ornl.gov/workshops/sns_hfir_users/posters/Laub_Chi-Square_Data_Fitting.pdf
"""

import numpy as np
from scipy.interpolate import splrep, splev
import pickle


# Read Quark-Gluon lattice data:
with open('QG_latticedata_WuB.p','rb') as file1:
    dicQG = pickle.load(file1)
file.close(open("QG_latticedata_WuB.p"))


chi2_lat = pickle.load(open('chi2_wubp.p', "rb"))
file.close(open('chi2_wubp.p'))



def chi2_sT3(p, data):                          # (p, *args) for fmin_l_bfgs_b
    """input:   p = [lambda_s, lambda_T],
                data = (s_array, T_array)
      output:   -(chi squared) for s/T^3 compared to lattice data

    ATTENTION: This fct is just used in file backbone_g_5_muocalc2_woVfit.py without any V or f fits
    """
    lambda_s, lambda_T = p
    s_array, T_array = data

    T_seq, s_seq = zip(*sorted(zip(T_array, s_array)))
    T_array = np.array(T_seq)
    s_array = np.array(s_seq)
    sT3_BH = lambda_s*s_array/(lambda_T*T_array)**3.0
    T_eval = lambda_T*T_array

    sT3_theo = splev(dicQG['T'], splrep(T_eval, sT3_BH))

    chi2 = np.sum( ( (sT3_theo-dicQG['sT3'])/dicQG['dsT3'] )**2.0 ) / (np.float(len(dicQG['T']))-2.0-1.0)

    return -chi2                                # ATTENTION:    -chi2 is returned for maximization!


def chi2_sT3_Lambda(p, data):                          # (p, *args) for fmin_l_bfgs_b
    """input:   p = [Lambda],
                data = (s_array, T_array)
      output:   -(chi squared) for s/T^3 compared to lattice data

    ATTENTION: This fct is just used in file backbone_g_5_muocalc2_woVfit.py without any V or f fits
    """
    Lambda = p[0]
    lambda_s = Lambda**3
    lambda_T = Lambda
    s_array, T_array = data

    T_seq, s_seq = zip(*sorted(zip(T_array, s_array)))
    T_array = np.array(T_seq)
    s_array = np.array(s_seq)
    sT3_BH = lambda_s*s_array/(lambda_T*T_array)**3.0
    T_eval = lambda_T*T_array

    sT3_theo = splev(dicQG['T'], splrep(T_eval, sT3_BH))

    chi2 = np.sum( ( (sT3_theo-dicQG['sT3'])/dicQG['dsT3'] )**2.0 ) / (np.float(len(dicQG['T']))-2.0-1.0)

    return -chi2


def chi2_suscep(p, data):
    """input:   p = [lambda_rhomu],
                data = (T_array, chi2hat_array, lambda_T)
      output:   -(chi squared) for quark susceptibility chi2hat compared to lattice data
    ATTENTION: This fct is just used in file backbone_g_5_muocalc2_woVfit.py without any V or f fits
    """
    lambda_rhomu = p
    T_array, chi2hat_array, lambda_T = data

    T_seq, chi2hat_seq = zip(*sorted(zip(T_array, chi2hat_array)))
    T_array = np.array(T_seq)
    chi2hat_array = np.array(chi2hat_seq)

    T_eval = lambda_T*T_array
    chi2hat_theo = splev(chi2_lat[:,0], splrep(T_eval, np.array(lambda_rhomu)/lambda_T**2.0*chi2hat_array)) # corrected missing lambda_T^2

    chi2 = np.sum( (chi2hat_theo-chi2_lat[:,1])**2 ) / np.float(len(chi2_lat[:,0]))

    return -chi2                                # ATTENTION:    -chi2 is returned for maximization!