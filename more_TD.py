import numpy as np
from scipy.integrate import quad, odeint
from scipy.interpolate import splrep, splev

from pylab import figure, plot, legend, show, semilogy, scatter, xlabel, ylabel, rc, axis, savefig, subplot, contour, contourf, colorbar, cm, axhline, axvline, getp

def e_and_I(TD_slice, p_raster):
    T_raster = TD_slice[:, 0]
    s_raster = TD_slice[:, 1]
    mu_raster = TD_slice[:, 2]
    rho_raster = TD_slice[:, 3]

    e_raster = s_raster*T_raster + rho_raster*mu_raster - p_raster
    I_raster = e_raster - 3.0*p_raster

    return [e_raster, I_raster]

def vsq(TD_slice, ivar):
    T_raster = TD_slice[:, 0]
    if T_raster[0] > T_raster[-1]:
        for k in range(0, 4):
            TD_slice[:, k] = TD_slice[:, k][::-1]

    T_raster = TD_slice[:, 0]
    s_raster = TD_slice[:, 1]
    mu_raster = TD_slice[:, 2]
    rho_raster = TD_slice[:, 3]
    print 'T:', T_raster
    print 's:', s_raster
    print 'mu:', mu_raster

    if ivar == 'T':
        vsq_t1 = 1.0/splev(np.log(s_raster),  splrep(np.log(s_raster),  np.log(T_raster)), der = 1)
        T_rho_tck = splrep(rho_raster, T_raster)
        drhodT = 1.0/splev(rho_raster, T_rho_tck, der = 1)
        if np.all(rho_raster != 0):
            vsq_raster = (vsq_t1 + mu_raster/s_raster*drhodT)**(-1)
        else:
            vsq_raster = (vsq_t1)**(-1)
    elif ivar == 'mu':
        vsq_t1 = 1.0/splev(np.log(rho_raster),  splrep(np.log(rho_raster),  np.log(mu_raster)), der = 1)
        mu_s_tck = splrep(s_raster, mu_raster)
        dsdmu = 1.0/splev(s_raster, mu_s_tck, der = 1)
        vsq_raster = (vsq_t1 + T_raster/rho_raster*dsdmu)**(-1)

    print 'vsq =', vsq_raster
    return vsq_raster


def chi2_calc(TD_slice): ##along a const mu line
    mu_raster = TD_slice[:, 2]
    rho_raster = TD_slice[:, 3]

    mu_rho_tck = splrep(rho_raster, mu_raster)
    chi2_raster = splev(rho_raster, mu_rho_tck, der = 1)**(-1)

    return chi2_raster