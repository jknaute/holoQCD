"""outsourced stuff"""

import numpy as np
from scipy.interpolate import splrep, splev
from scipy.optimize import brentq
import numpy.ma as ma
from pylab import *

model_type = 'VRY_4'
T_CEP = 111.5
if model_type == 'VRY_2':
    mu_CEP = 987.5
if model_type == 'VRY_4':
    mu_CEP = 611.5



def pressure_like(S, T):
    """dp = s(T)dT, n(mu)=const"""

    p = np.zeros(len(S))
    for i in range(1, len(S)):
        dT = T[i] - T[i-1]
        p[i] = p[i-1] + 0.5*(S[i-1] + S[i])*dT  # trapezoidal method
    p[0] = p[1]

    return p


def nearest_neighbor_sort(x, y):
    """Sort (x,y) by nearest neighbors to get multivalued s-shape form in line plot
    input:  x,y arrays are ordered by increasing x;
            first and last pt must be correct
    """

    x_sorted = []
    y_sorted = []
    xy_used = []
    xy_possible = range(len(x))

    xy_used.append(0)
    xy_possible.remove(0)
    x_sorted.append(x[0])
    y_sorted.append(y[0])

    while len(xy_used) < len(x):
        d_following = np.sqrt((x[xy_possible]-x[xy_used[-1]])**2 + (y[xy_possible]-y[xy_used[-1]])**2)
        d_min_pos = xy_possible[d_following.argmin()]

        xy_used.append(d_min_pos)
        xy_possible.remove(d_min_pos)
        x_sorted.append(x[d_min_pos])
        y_sorted.append(y[d_min_pos])
        #print 'd_min_pos, xy_used, xy_possible: ', d_min_pos, xy_used, xy_possible

    x_sorted = np.array(x_sorted)
    y_sorted = np.array(y_sorted)
    #print 'lengths x, x_sorted: ', len(x), len(x_sorted)

    return [x_sorted, y_sorted]

#X = np.array([1,2,2,2,3,3,3,4,5])
#Y = np.array([5,5,3,1,5,3,1,1,1])
#X, Y = nearest_neighbor_sort(X,Y)

#plot(X,Y)
#axis([0,5,0,6])
#show()


def tdv_PT_finder(tdv, ltp_tck, htp_tck):
    """Calculation of mu_c for phase transition"""
    return splev(tdv, htp_tck) - splev(tdv, ltp_tck)


def get_phases(p, T):
    """get phase type from function p(T) and possible Tc"""

    PT_type = 'ifl'
    Tc = 'na'
    ltp_Tmax = 'na'   # reversal pts in s-shape
    htp_Tmin = 'na'
    ltp_maxind = 'na'
    htp_minind = 'na'

    for j in range(1, len(T)):
        if T[j] < T[j-1]:
            PT_type = '1st'
            ltp_maxind = j-1
            break
    if PT_type=='1st':
        for j in range(1, len(T))[::-1]:
            if T[j] < T[j-1]:
                htp_minind = j
                break
        #print 'ltp_maxind, htp_minind: ', ltp_maxind, htp_minind

        p_ltp_tck = splrep(T[0:ltp_maxind], p[0:ltp_maxind])
        p_htp_tck = splrep(T[htp_minind:], p[htp_minind:])
        Tc = brentq(tdv_PT_finder, T[htp_minind], T[ltp_maxind], xtol=1e-12, rtol=1e-10, args = (p_ltp_tck, p_htp_tck))

        ltp_Tmax = T[ltp_maxind]
        htp_Tmin = T[htp_minind]

    return [PT_type, Tc, htp_Tmin, ltp_Tmax, ltp_maxind, htp_minind]


def mask_grid(S_matr, TD_gr, Tc_muT, phase):
    """
    Masking S_HEE matrix:
    pts outside plotting range and multivalued region
    """
    S_matr_m = ma.asarray(S_matr)   # masked

    if model_type == 'VRY_2':
        T_min = 50.0
        T_max = 200.0
        mu_min = 0.0
        mu_max = 1450.0
    elif model_type == 'VRY_4':
        T_min = 0.5*T_CEP
        T_max = 1.8*T_CEP
        mu_min = 0.0
        mu_max = 1.5*mu_CEP

    Tc_muT_tck = splrep(Tc_muT[0], Tc_muT[1])
    S_up_tck = splrep(Tc_muT[0], Tc_muT[4])
    S_mid_tck = splrep(Tc_muT[0], Tc_muT[5])
    S_down_tck = splrep(Tc_muT[0], Tc_muT[6])

    #plot(Tc_muT[0], splev(Tc_muT[0], S_up_tck))
    #plot(Tc_muT[0], splev(Tc_muT[0], S_mid_tck))
    #plot(Tc_muT[0], splev(Tc_muT[0], S_down_tck))
    #show()

    for i in range(0, len(S_matr[:,0])):
        for j in range(0, len(S_matr[0,:])):
            T = TD_gr[i,j,0]
            mu = TD_gr[i,j,2]
            S = np.log(S_matr_m[i,j])   # final S_HEE refers to log values

            if T > T_max*1.05 or mu > mu_max*1.1:
                S_matr_m[i,j] = ma.masked
            else:
                if mu > mu_CEP and T < T_CEP:   # FOPT region
                    T_proj_phasecont = splev(mu, Tc_muT_tck)
                    S_up_proj = splev(mu, S_up_tck)
                    S_mid_proj = splev(mu, S_mid_tck)
                    S_down_proj = splev(mu, S_down_tck)

                    if T > T_proj_phasecont and S < S_up_proj and S >= S_mid_proj and (phase=='ltp' or phase=='all'):   # ltp instable branch above Tc
                        S_matr_m[i,j] = ma.masked
                        print 'masking T = %2.2f, mu = %2.2f, S_HEE = %2.2f' %(T, mu, S)

                    if T < T_proj_phasecont and S > S_down_proj and S <= S_mid_proj and (phase=='htp' or phase=='all'): # htp instable branch below Tc
                        S_matr_m[i,j] = ma.masked
                        print 'masking T = %2.2f, mu = %2.2f, S_HEE = %2.2f' %(T, mu, S)

    return S_matr_m


def fct_scalar(x, y_tck):
    """Calculation of inflection pt via 2nd derivative """
    return splev(x, y_tck, der=2)


def critical_behavior_hee(T, S):
    """
    Calculation of the critical exponent alpha for C_mu (heat capacity at constant mu):
        C_mu ~ (T - T_CEP)^-alpha
    for HEE
    """
    for i in range(1, len(S)):
        if S[::-1][i] < S[::-1][i-1]:
            ind_max = i-1
            break
    print 'ind_max = ', ind_max     # where turnaround starts that cannot be splined


    TS_tck = splrep(S[::-1][:ind_max], T[::-1][:ind_max]) # = T(S)
    dTdS = splev(S[::-1][:ind_max], TS_tck, der=1)
    C_mu = T * splev(T, splrep(T, S), der=1)
    Sc = brentq(fct_scalar, 20.0, 24.0, xtol=1e-12, rtol=1e-10, args=(TS_tck,)) # S at inflection pt
    Tc = splev(Sc, TS_tck)
    print 'Sc, Tc: ', Sc, Tc
    ind_CEP = np.amax(np.where(T<=Tc))


    figure(100) # log(T) - log(S)
    lnS_below = np.log(np.abs(S-Sc))[ind_CEP-15:ind_CEP]      # for T < T_CEP
    lnT_below = np.log(np.abs(T-Tc))[ind_CEP-15:ind_CEP]
    lnS_above = np.log(np.abs(S-Sc))[ind_CEP+2:ind_CEP+14]    # for T > T_CEP
    lnT_above = np.log(np.abs(T-Tc))[ind_CEP+2:ind_CEP+14]

    linfit_below = np.polyfit(lnS_below, lnT_below, 1)
    linfct_below = np.poly1d(linfit_below)
    linfit_above = np.polyfit(lnS_above, lnT_above, 1)
    linfct_above = np.poly1d(linfit_above)
    print '<T_CEP hee: linear fit, alpha = ', linfit_below, -(1.0/linfit_below[0]-1)
    print '>T_CEP hee: linear fit, alpha = ', linfit_above, -(1.0/linfit_above[0]-1)

    plot(lnS_below, lnT_below, ls='', marker='s', c='b', label=r'$T < T_{CEP}$')
    plot(lnS_above, lnT_above, ls='', marker='s', c='r', label=r'$T > T_{CEP}$')
    plot(lnS_below, linfct_below(lnS_below), c='b')
    plot(lnS_above, linfct_above(lnS_above), c='r')


    figure(101) # T(S)
    plot(S, T)
    plot(Sc, Tc, ls='', marker='s', c='r')
    plot(S[::-1][:ind_max], dTdS)


    figure(102) # C_mu(T)
    plot(T, C_mu)


    figure(103) # log(C_mu) - log(T)
    lnC_below = np.log(C_mu)[ind_CEP-15:ind_CEP]
    lnC_above = np.log(C_mu)[ind_CEP+2:ind_CEP+14]
    plot(lnT_below, lnC_below, ls='', marker='s', c='b')
    plot(lnT_above, lnC_above, ls='', marker='s', c='r')

    return 1


def critical_behavior_TD(T, S):
    """
    Calculation of the critical exponent alpha for C_mu (heat capacity at constant mu):
        C_mu ~ (T - T_CEP)^-alpha
    for thermodynamic entropy
    """
    if T[0] > T[-1]:
        T = T[::-1]
        S = S[::-1]

    TS_tck = splrep(S, T)   # = T(S)
    dTdS = splev(S, TS_tck, der=1)
    C_mu = T * splev(T, splrep(T, S), der=1)
    Sc = brentq(fct_scalar, 0.7e7, 1.0e7, xtol=1e-12, rtol=1e-10, args=(TS_tck,)) # S at inflection pt
    Tc = splev(Sc, TS_tck)
    print 'Sc, Tc: ', Sc, Tc
    ind_CEP = np.amax(np.where(T<=Tc))


    figure(100) # log(T) - log(S)
    lnS_below = np.log(np.abs(S-Sc))[ind_CEP-15:ind_CEP]      # for T < T_CEP
    lnT_below = np.log(np.abs(T-Tc))[ind_CEP-15:ind_CEP]
    lnS_above = np.log(np.abs(S-Sc))[ind_CEP+2:ind_CEP+14]    # for T > T_CEP
    lnT_above = np.log(np.abs(T-Tc))[ind_CEP+2:ind_CEP+14]

    linfit_below = np.polyfit(lnS_below, lnT_below, 1)
    linfct_below = np.poly1d(linfit_below)
    linfit_above = np.polyfit(lnS_above, lnT_above, 1)
    linfct_above = np.poly1d(linfit_above)
    print '<T_CEP TD: linear fit, alpha = ', linfit_below, -(1.0/linfit_below[0]-1)
    print '>T_CEP TD: linear fit, alpha = ', linfit_above, -(1.0/linfit_above[0]-1)

    plot(lnS_below, lnT_below, ls='', marker='s', c='b', label='_nolegend_')
    plot(lnS_above, lnT_above, ls='', marker='s', c='r')
    plot(lnS_below, linfct_below(lnS_below), c='b')
    plot(lnS_above, linfct_above(lnS_above), c='r')


    figure(101) # T(S)
    plot(S, T)
    plot(S[ind_CEP+2:ind_CEP+14], T[ind_CEP+2:ind_CEP+14], ls='', marker='s', c='g')
    plot(Sc, Tc, ls='', marker='s', c='r')
    plot(S, dTdS)


    figure(102) # C_mu(T)
    plot(T, C_mu)


    figure(103) # log(C_mu) - log(T)
    lnC_below = np.log(C_mu)[ind_CEP-15:ind_CEP]
    lnC_above = np.log(C_mu)[ind_CEP+2:ind_CEP+14]
    plot(lnT_below, lnC_below, ls='', marker='s', c='b')
    plot(lnT_above, lnC_above, ls='', marker='s', c='r')

    return 1








































