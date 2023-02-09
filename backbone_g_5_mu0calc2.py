"""
Contains V(phi) and f(phi) Parameter fit
"""

import numpy as np
from scipy.integrate import quad, odeint
from scipy.interpolate import splrep, splev

from amoba import amoeba
from scipy.optimize import fmin_l_bfgs_b

from Vtypes import Vs, dVs, dlogVs
from ftypes import fs, dfs
from H_expansions_G import get_horizon_expansions
from args_and_lambds import args_dic, lambdas_dic

from rasterizer import rasterize
from time import time
import pickle

from pylab import figure, plot, legend, show, semilogy, loglog, axis, subplot, xlabel, ylabel, rc, savefig, errorbar
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FixedLocator, LinearLocator
from matplotlib.transforms import Bbox

from nice_tcks import nice_ticks
from p_calcer4 import p_calc_line
from fmg_TDprocess import TD_scale_isen
from more_TD import e_and_I, vsq

from amoba import amoeba


linew = 3
rc('font', size = 20) #fontsize of axis labels (numbers)
rc('axes', labelsize = 22, lw = linew) #fontsize of axis labels (symbols)
rc('lines', mew = 2, lw = linew, markeredgewidth = 2)
rc('patch', ec = 'k')
rc('xtick.major', pad = 7)
rc('ytick.major', pad = 7)

MS = 5      # marker size
MEW = 1     # marker edge width

#a__tol = 1e-80
r__tol = 1e-13
L = 1.0
mxstp = 4000

def V(phi, *args):
    return Vs[args[0]](phi, *args[1])

def dV_dphi(phi, *args):
    return dVs[args[0]](phi, *args[1])

def f(phi, *args):
    return fs[args[0]](phi, *args[1])

def df_dphi(phi, *args):
    return dfs[args[0]](phi, *args[1])

def EFEs_G_full(y, r, V_args, f_args):
    A = y[0]
    dA = y[1]
    phi = y[2]
    dphi = y[3]
    h = y[4]
    dh = y[5]
    #Phi = y[6]
    dPhi = y[7]

    f_val = f(phi, *f_args)
    df_dphi_val = df_dphi(phi, *f_args)

    ddA = - 1.0/6.0*dphi**2.0
    ddh = - 4.0*dA*dh + np.exp(-2.0*A)*f_val*dPhi**2.0
    ddPhi = - 2.0*dA*dPhi - df_dphi_val/f_val*dphi*dPhi
    ddphi = - (4.0*dA + dh/h)*dphi + 1.0/h*(dV_dphi(phi, *V_args) - 1.0/2.0*np.exp(-2.0*A)*df_dphi_val*dPhi**2.0)

    return [dA, ddA, dphi, ddphi, dh, ddh, dPhi, ddPhi]

def EFEs_G_Alogphi(y, r, V_args, h_val):
    #A = y[0]
    dA = y[1]
    psi = y[2]
    dpsi = y[3]

    ddA = - 1.0/6.0*dpsi**2.0*np.exp(2.0*psi)
    ddpsi = - dpsi**2.0 - 4.0*dA*dpsi + np.exp(-psi)/h_val*dV_dphi(np.exp(psi), *V_args)

    return [dA, ddA, dpsi, ddpsi]

def zec_check(sol, V_args, f_args):
    V_raster = rasterize(V, sol['phi'], *V_args)[1]
    f_raster = rasterize(f, sol['phi'], *f_args)[1]
    zec = sol['h']*(24.0*sol['dA']**2.0 - sol['dphi']**2.0) + 6.0*sol['dA']*sol['dh'] + 2.0*V_raster + np.exp(-2.0*sol['A'])*sol['dPhi']**2.0*f_raster
    return zec

def QN_check(sol, f_args):
    f_raster = rasterize(f, sol['phi'], *f_args)[1]
    Q_N = np.exp(2.0*sol['A'])*(np.exp(2.0*sol['A'])*sol['dh'] - f_raster*sol['Phi']*sol['dPhi'])
    return Q_N

def QG_check(sol, f_args):
    f_raster = rasterize(f, sol['phi'], *f_args)[1]
    Q_G = np.exp(2.0*sol['A'])*f_raster*sol['dPhi']
    return Q_G

def get_UV_region(solution, mode, cutoff):
    solution_UV = {}
    r_raster = solution['r']
    for func in solution.keys():
        if mode == 'r_value':
            solution_UV.update({func:np.compress(r_raster > cutoff, solution[func])})
        elif mode == 'r_index':
            solution_UV.update({func:solution[func][cutoff:]})
    return solution_UV

def fit_A_UV(p, data):
    r_raster = data[0]
    A_raster = data[1]

    Am1f = p[0]
    A0f = p[1]

    A_UV = Am1f*r_raster/L + A0f

    chiq = - np.log(np.sum((A_raster - A_UV)**2.0)/np.float(len(r_raster)))
    #print chiq
    return chiq

def get_UV_coeffs(solution_UV, f_args, Delta):
    h0f_init = solution_UV['h'][-1]
    A0f_init = solution_UV['A'][-1] - 1.0/np.sqrt(h0f_init)*solution_UV['r'][-1]
    #print 'h0f_init = %2.12f, A0f_init = %2.12f' %(h0f_init, A0f_init)
    Afit = amoeba([1.0/np.sqrt(h0f_init), A0f_init], [A0f_init/50.0, h0f_init/50.0], fit_A_UV, xtolerance=1e-10, ftolerance=1e-8, data = [solution_UV['r'], solution_UV['A']])[0]

    h0f = h0f_init
    A0f = Afit[1]
    Phi0f = solution_UV['Phi'].mean()
    log_phi_A = (np.log(solution_UV['phi']) + (4.0 - Delta)*(1.0/np.sqrt(h0f)*solution_UV['r']/L + A0f)).mean()
    phi_A = np.exp(log_phi_A)

    if solution_UV['dh'][-1] == 0:
        print 'Am1f and h0f:', 1.0/Afit[0]**2.0, h0f_init, 1.0/Afit[0]**2.0/h0f_init
        print 'h0f = %2.12f, Am1f = %2.12f, Phi0f = %2.12f, phi_A = %2.12f, A0f = %2.12f' %(h0f, 1.0/np.sqrt(h0f), Phi0f, phi_A, A0f)

    return ([h0f, Phi0f, phi_A, A0f, Afit[0]])

def EFEs_solver_pw(phi_0, Phi_1, eps, r_mid, V_args, f_args, a_tol, r_tol, Delta):
    order = 5
    A_1 = -1.0/3.0*L*(1.0/2.0*f(phi_0, *f_args)*Phi_1**2.0 + V(phi_0, *V_args))
    eps = 1e-3/(phi_0*A_1)

    ICs_h = get_horizon_expansions(eps, order, phi_0, Phi_1, V_args, f_args)[0]
    #print 'A_1 =', A_1, 'eps =', eps, 'r_mid =', r_mid
    #print 'ICs_h =', ICs_h

    r_raster = np.hstack((np.linspace(eps, r_mid/20.0, 2000), np.linspace(r_mid/20.0, r_mid, 2001)[1:]))
    #r_raster = np.hstack((np.linspace(eps**0.5, (r_mid/40.0)**0.5, 4000)**2.0, np.linspace(r_mid/40.0, r_mid, 2001)[1:]))

    solution_1_wmsg = odeint(EFEs_G_full, ICs_h, r_raster, args = (V_args, f_args), atol = 1e-50, rtol = r_tol, full_output = True, mxstep = mxstp)
    #print solution_1_wmsg[1]['message']
    if solution_1_wmsg[1]['message'] != 'Integration successful.':
        return 0 # raise ValueError
    else:
        solution_1 = solution_1_wmsg[0]

    sol1 = {'r':r_raster, 'A':solution_1[:, 0], 'dA':solution_1[:, 1], 'phi':solution_1[:, 2], 'dphi':solution_1[:, 3], 'h':solution_1[:, 4], 'dh':solution_1[:, 5], 'Phi':solution_1[:, 6], 'dPhi':solution_1[:, 7]}
    #print solution_1[:, 2][::100]
    if np.any(sol1['phi'] < 0):
        print 'negative phi, discarding solution...'
        return 0 # raise ValueError

    if Phi_1 == 0:
        r_match = np.compress(sol1['dh']/sol1['h'] > 1e-30, r_raster)[-1]
        r_ind_match = len(np.compress(r_raster <= r_match, r_raster)) - 1
        print 'switching to 0T eqns at', r_match
    else:
        if len(np.compress(sol1['dPhi']/sol1['Phi'] > 1e-30, r_raster)) == 0:
            return 0    # for avoiding errors, discard sol
        r_match = np.compress(sol1['dPhi']/sol1['Phi'] > 1e-30, r_raster)[-1]
        r_ind_match = len(np.compress(r_raster <= r_match, r_raster)) - 1
        print 'switching to 0T eqns at', r_match

    #print 'r mid check:', r_mid, r_match, r_mid/40.0
    if r_mid == r_match:
        print 'WARNING'

    ICs_m = [sol1['A'][r_ind_match], sol1['dA'][r_ind_match], np.log(sol1['phi'][r_ind_match]), sol1['dphi'][r_ind_match]/sol1['phi'][r_ind_match]]
    #print sol1['phi'][-40:], sol1['dphi'][-40:]
    #print ICs_m

    UV_coeffs_m = get_UV_coeffs(get_UV_region(sol1, 'r_index', -20), f_args, Delta)
    h0m = UV_coeffs_m[0]
    phi_Am = UV_coeffs_m[2]
    A0m = UV_coeffs_m[3]
    Am1m = UV_coeffs_m[4]

    psi_min = 120.0
    r_inf = np.sqrt(h0m)/(4.0 - Delta)*(psi_min + np.log(phi_Am) - (4.0 - Delta)*A0m)
    r_inf = 1.0/(Am1m*(4.0 - Delta))*(psi_min + np.log(phi_Am) - (4.0 - Delta)*A0m)
    #print 'r_inf =', r_inf

    r_raster2 = np.linspace(r_match, r_inf, 4000)
    h_val = sol1['h'][r_ind_match]
    Phi_val = sol1['Phi'][r_ind_match]
    solution_2 = odeint(EFEs_G_Alogphi, ICs_m, r_raster2, args = (V_args, h_val), atol = 1e-20, rtol = r_tol, mxstep = mxstp)

    sol2 = {'r':r_raster2, 'A':solution_2[:, 0], 'dA':solution_2[:, 1], 'phi':np.exp(solution_2[:, 2]), 'dphi':solution_2[:, 3]*np.exp(solution_2[:, 2]), 'h':h_val*np.ones(len(r_raster2)), 'dh':np.zeros(len(r_raster2)), 'Phi':Phi_val*np.ones(len(r_raster2)), 'dPhi':np.zeros(len(r_raster2))}

    sol_full = {}
    for key in sol1.keys():
        sol_full.update({key:np.hstack((sol1[key][:r_ind_match], sol2[key][1:]))})

    return sol_full


def TD_from_UV_coeffs(UV_coeffs, Q_G, Delta, kappa_5):
    h0f = UV_coeffs[0]
    Phi0f = UV_coeffs[1]
    phi_A = UV_coeffs[2]

    T = (L*4.0*np.pi*np.sqrt(h0f)*phi_A**(1.0/(4.0 - Delta)))**(-1.0)
    mu = 4.0*np.pi*Phi0f*T
    s = 2.0*np.pi/kappa_5**2.0/(phi_A**(3.0/(4.0 - Delta)))
    rho = L*Q_G/(4.0*np.pi)*s # correct missing f(0) !!!
    print np.array([T, s, mu, rho])

    return np.array([T, s, mu, rho])

def chi_2_atmu0_from_UV_coeffs(UV_coeffs, Q_G, kappa_5):
    h0f = UV_coeffs[0]
    Phi0f = UV_coeffs[1]
    chi_2 = 8.0*np.pi**2.0*L**4.0/kappa_5**2.0*Q_G*h0f**(3.0/2.0)*Phi0f

    return chi_2

def Delta_from_Vargs(V_args):
    if V_args[0] == 'V_I' or V_args[0] == 'V_VI':
        mq = 2.0*V_args[1][1] - 12.0*V_args[1][0]**2.0 #check
        Delta = 2.0 + np.sqrt(4.0 + mq)
    return Delta

##### full TD grid #####
def TD_calc(phi0_min, phi0_max, phi0_pts, Phi1_min, Phi1_pts, V_args, f_args, r_mid_0, eps, kappa_5):
    log_phi0 = np.linspace(np.log(phi0_min), np.log(phi0_max), phi0_pts)
    phi0_raster = np.exp(log_phi0)
    #print log_phi0, phi0_raster

    Phi1_max_raster = np.array([np.sqrt(-2.0*V(phi_0, *V_args)/f(phi_0, *f_args)) for phi_0 in phi0_raster]) # maximum value in order to have positive A_1 thus monotonous dA/dr

    phi_Phi_grid = np.zeros((phi0_pts, Phi1_pts, 2))
    r_mids = np.zeros((phi0_pts, Phi1_pts))
    TD_grid = np.zeros((phi0_pts, Phi1_pts, 4))
    TD_dic = {}

    Delta = Delta_from_Vargs(V_args)
    a_tol = 1e-80 #10.0**( - (4.0*(4.0 - Delta)*r_inf) - 16)
    print 'Delta =', Delta, 'a_tol =', a_tol

    r_mids[0, 0] = r_mid_0
    for i in range(0, len(phi0_raster)):
        r_mid = r_mid_0
        if i > 1:
            r_mid = r_mids[i - 1, 0]
        phi_0 = phi0_raster[i]
        Phi1_max = Phi1_max_raster[i]
        print 20*'#'
        print 'Phi1_max =', Phi1_max
        Phi1r_raster = np.linspace(0.0, Phi1r_max, Phi1_pts) # attention: global Phi1r_max
        Phi1_raster = Phi1r_raster*Phi1_max
        if Phi1_pts == 1:
            Phi1_raster = np.array([Phi1_min])

        for j in range(0, len(Phi1_raster)):
            Phi_1 = Phi1_raster[j]
            print 'phi_0 = %2.5f, log phi_0 = %2.5f, Phi_1 = %2.5f, Phi_1/Phi_1_max = %2.5f' %(phi_0, np.log(phi_0), Phi_1, Phi_1/Phi1_max)
            Q_G = f(phi_0, *f_args)*Phi_1
            print 'r_mid =', r_mid
            print 'Q_G =', Q_G
            try:
                metric_sol = EFEs_solver_pw(phi_0, Phi_1, eps, r_mid, V_args, f_args, a_tol, r__tol, Delta)
            except ValueError:
                print 'Initial values phi_0 = %2.5f Phi_1/Phi_1_max = %2.5f' %(phi_0, Phi_1/Phi1_max), 'fail to produce a good black hole solution. Switching to next phi_0 value'
                #show()
                break

            phi_Phi_grid[i, j, :] = np.array([phi_0, Phi_1/Phi1_max])
            r_mids[i, j] = metric_sol['r'][len(np.compress(metric_sol['dh'] > 0, metric_sol['r']))]
            if Phi_1 == 0 and len(Phi1_raster) == 1:
                r_mid = r_mids[i, j]
            else:
                if Phi_1/Phi1_max < 0.75:
                    r_mid = metric_sol['r'][-1]/30.0*(1.0 + 2.0*Phi_1/Phi1_max)
                else:
                    r_mid = metric_sol['r'][-1]/30.0*(1.0 + 10.0*Phi_1/Phi1_max)

            solution_UV = get_UV_region(metric_sol, 'r_index', -50)
            UV_coeffs = get_UV_coeffs(solution_UV, f_args, Delta)
            TD_grid[i, j, :] = TD_from_UV_coeffs(UV_coeffs, Q_G, Delta, kappa_5)

            TD_dic.update({str(i)+'_'+str(j):{'phi_0':phi_0, 'Phi_1':Phi_1, 'T':TD_grid[i, j, 0], 's':TD_grid[i, j, 1], 'mu':TD_grid[i, j, 2], 'rho':TD_grid[i, j, 3]}})

            print 'T = %2.5f, s = %2.5f, mu = %2.5f, rho = %2.5f' %tuple([TD_grid[i, j, k] for k in range(0, 4)])
            print 8*'#'

    TD_dic_full = {'V_args':V_args, 'f_args':f_args, 'phi0_raster':phi0_raster, 'Phi1_raster':Phi1_raster, 'Phi1r_raster':Phi1r_raster,'phi_Phi':phi_Phi_grid, 'TD':TD_grid, 'phi0_pts':phi0_pts, 'Phi1_pts':Phi1_pts}

    return [TD_grid, phi_Phi_grid, TD_dic_full]

##### isentropes #####
def phi0_range_isentrope(epb_value, phi0_min, phi0_max, phi0_pts, V_args, f_args):
    phi0_raster = np.linspace(phi0_min, phi0_max, phi0_pts)
    Phi1_max_raster = np.array([np.sqrt(-2.0*V(phi_0, *V_args)/f(phi_0, *f_args)) for phi_0 in phi0_raster])
    Q_G = 4.0*np.pi/(L*epb_value)
    Phi1_raster = Q_G/np.array([f(phi_0, *f_args) for phi_0 in phi0_raster])
    Phi1r_raster = Phi1_raster/Phi1_max_raster
    #print Phi1r_raster
    phi0_comp = np.compress(Phi1r_raster < 0.95, phi0_raster)
    #print phi0_comp
    #print buh
    return [phi0_comp[0], phi0_comp[-1]]

def isentrope_calc(epb_value, phi0_min, phi0_max, phi0_pts, V_args, f_args, r_mid_0, eps, kappa_5):
    phi0_raster = np.linspace(phi0_min, phi0_max, phi0_pts)
    Phi1_max_raster = np.array([np.sqrt(-2.0*V(phi_0, *V_args)/f(phi_0, *f_args)) for phi_0 in phi0_raster]) # maximum value in order to have positive A_1 thus monotonous dA/dr

    Q_G = 4.0*np.pi/(L*epb_value)
    print 'Calculating isentrope for rho/s = %2.8f => Q_G = %2.8f' %(epb_value*epb_lamb, Q_G)

    Phi1_raster = Q_G/np.array([f(phi_0, *f_args) for phi_0 in phi0_raster])

    print 'Corresponding Phi_1 vals:', Phi1_raster
    print 'Phi_1 maxs:', Phi1_max_raster
    print 'Phi_1/Phi_1_max:', Phi1_raster/Phi1_max_raster

    r_mids = np.zeros(phi0_pts)
    TD_isentrope = np.zeros((phi0_pts, 4))
    phi_Phi_isentrope = np.zeros((phi0_pts, 2))
    TD_dic = {}

    goodpt_raster = np.ones(phi0_pts)

    a_tol = 1e-80
    Delta = Delta_from_Vargs(V_args)
    print 'Delta =', Delta, 'a_tol =', a_tol

    r_mids[0] = r_mid_0
    for i in range(0, len(phi0_raster)):
        print i
        #goodpt = 1
        #r_mid = r_mid
        if i < 1:
            r_mid = r_mid_0

        phi_0 = phi0_raster[i]
        Phi_1 = Phi1_raster[i]
        Phi1_max = Phi1_max_raster[i]
        phi_Phi_isentrope[i, :] = np.array([phi_0, Phi_1/Phi1_max])

        print 'phi_0 = %2.5f, log phi_0 = %2.5f, Phi_1 = %2.5f, Phi_1/Phi_1_max = %2.5f' %(phi_0, np.log(phi_0), Phi_1, Phi_1/Phi1_max)

        print 'r_mid =', r_mid
        try:
            metric_sol = EFEs_solver_pw(phi_0, Phi_1, eps, r_mid, V_args, f_args, a_tol, r__tol, Delta)
        except ValueError:
            print 'Initial values phi_0 = %2.5f Phi_1/Phi_1_max = %2.5f' %(phi_0, Phi_1/Phi1_max), 'fail to produce a good black hole solution. Switching to next phi_0 value'
            goodpt_raster[i] = 0

            if i > 1 and goodpt_raster[i - 1] == 1:
                goodpt_raster[i:] = np.zeros(len(goodpt_raster[i:]))
                break

        if goodpt_raster[i]:
            r_mids[i] = metric_sol['r'][len(np.compress(metric_sol['dh'] > 0, metric_sol['r']))]
            if Phi_1 == 0 and len(Phi1_raster) == 1:
                r_mid = r_mids[i]
            else:
                if Phi_1/Phi1_max < 0.60:
                    r_mid = metric_sol['r'][-1]/30.0*(1.0 + 2.0*Phi_1/Phi1_max)
                else:
                    r_mid = metric_sol['r'][-1]/20.0*(1.0 + 20.0*Phi_1/Phi1_max)

            solution_UV = get_UV_region(metric_sol, 'r_index', -50)
            UV_coeffs = get_UV_coeffs(solution_UV, f_args, Delta)
            TD_isentrope[i, :] = TD_from_UV_coeffs(UV_coeffs, Q_G, Delta, kappa_5)

            TD_dic.update({str(i):{'phi_0':phi_0, 'Phi_1':Phi_1, 'T':TD_isentrope[i, 0], 's':TD_isentrope[i, 1], 'mu':TD_isentrope[i, 2], 'rho':TD_isentrope[i, 3]}})

            print 'T = %2.5f, s = %2.5f, mu = %2.5f, rho = %2.5f' %tuple([TD_isentrope[i, k] for k in range(0, 4)])
            print 's/rho =', TD_isentrope[i, 1]/TD_isentrope[i, 3]*epb_lamb
            print 8*'#'

    isentrope_dic_full = {'epb':epb_value, 'Q_G':Q_G, 'V_args':V_args, 'f_args':f_args, 'phi0_raster':phi0_raster, 'Phi1_raster':Phi1_raster, 'phi_Phi':phi_Phi_isentrope, 'TD':TD_isentrope, 'phi0_pts':phi0_pts}

    return [TD_isentrope, phi_Phi_isentrope, isentrope_dic_full]

def TD_calc_pointwise(phi_0, Phi_1r, V_args, f_args, r_mid):
    Phi_1_max = np.sqrt(-2.0*V(phi_0, *V_args)/f(phi_0, *f_args))
    Phi_1 = Phi_1r*Phi_1_max
    Q_G = f(phi_0, *f_args)*Phi_1

    Delta = Delta_from_Vargs(V_args)
    a_tol = 1e-80 #10.0**( - (4.0*(4.0 - Delta)*r_inf) - 16)
    eps = 1e-6
    #print 'Delta =', Delta, 'a_tol =', a_tol

    metric_sol = EFEs_solver_pw(phi_0, Phi_1, eps, r_mid, V_args, f_args, a_tol, r__tol, Delta)
    if metric_sol == 0:
        return 0
    solution_UV = get_UV_region(metric_sol, 'r_index', -50)
    UV_coeffs = get_UV_coeffs(solution_UV, f_args, Delta)
    TD_point = TD_from_UV_coeffs(UV_coeffs, Q_G, Delta, kappa_5)
    #print TD_point

    return [TD_point, metric_sol]

def chi2_integrand(r, tck):
    return np.exp(splev(r, tck[0]))

def chi2_mu0_calc_pt(TD_point, metric_sol, f_args, i, return_integral):
    r_raster = metric_sol['r']
    A_raster = metric_sol['A']
    phi_raster = metric_sol['phi']
    f_raster = rasterize(f, phi_raster, *f_args)[1]

    log_integrand = -2.0*A_raster - np.log(f_raster)
    r_raster = np.compress(log_integrand > - 200.0, r_raster)
    log_integrand = np.compress(log_integrand > - 200.0, log_integrand)

    log_integrand_tck = splrep(r_raster, log_integrand, k = 3)
    integral = quad(chi2_integrand, r_raster[0], r_raster[-1], args = [log_integrand_tck], limit = 400, epsrel = 1e-10, epsabs = 1e-10)[0]

    #print 'integrating for chi/T^2 up to', r_raster[-1]
    #print np.compress(r_raster < r_raster[-1]*2.0/3.0, np.exp(log_integrand))
    #print len(r_raster)
    #print np.exp(-2.0*A_raster - np.log(f_raster))[::40]

    # lw = 1
    # ls = 'solid'
    # if i == 65:
    #     lw = 2
    #     ls = 'dashed'
    # figure(20)
    # plot(r_raster, np.exp(log_integrand), lw = lw, ls = ls)
    # figure(21)
    # rr2 = np.linspace(r_raster[0], r_raster[-1], 20000)
    # plot(r_raster, log_integrand, lw = lw, ls = ls)
    # plot(rr2, splev(rr2, log_integrand_tck), lw = lw, ls = ls)

    chi2T2 = L/(16.0*np.pi**2.0)*TD_point[1]/TD_point[0]**3.0/integral
    print 'chi2T2 =', chi2T2

    if not return_integral:
        return chi2T2
    else:
        return [chi2T2, integral]


def TD_calc_mu0(phi0_min, phi0_max, phi0_pts, V_args, f_args):
    log_phi0 = np.linspace(np.log(phi0_min), np.log(phi0_max), phi0_pts)
    phi0_raster = np.exp(log_phi0)

    TD_T_axis = np.zeros((phi0_pts, 4))
    chiT2_raster = np.zeros(phi0_pts)

    return_integral = 1
    integral_raster = np.zeros(phi0_pts)
    metric_sols_list = [None]*phi0_pts

    r_mid = 12.0

    for i in range(0, len(phi0_raster)):
        phi_0 = phi0_raster[i]
        msTD = TD_calc_pointwise(phi_0, 0, V_args, f_args, r_mid)

        if msTD == 0: # try to disregard pts with no solution from integration
            return 0

            #print 'phi0 value disregarded: ', phi_0, phi0_pts, len(phi0_raster)
            #TD_T_axis = np.delete(TD_T_axis, i, 0)
            #chiT2_raster = np.delete(chiT2_raster, i)
            #integral_raster = np.delete(integral_raster, i)
            #metric_sols_list = np.delete(metric_sols_list, i)
            #phi0_raster = np.delete(phi0_raster, i)
            #print 'len new phi0_raster: ', len(phi0_raster)
            #continue # go to next phi0 value in array
        else:
            #print i, phi_0, msTD[0]#, TD_T_axis[i]
            TD_T_axis[i, :] = msTD[0]
            metric_sol = msTD[1]
            r_mid = metric_sol['r'][len(np.compress(metric_sol['dh'] > 0, metric_sol['r']))]

            chiT2calc = chi2_mu0_calc_pt(msTD[0], metric_sol, f_args, i, return_integral)
            chiT2_raster[i] = chiT2calc[0]
            integral_raster[i] = chiT2calc[1]
            metric_sols_list[i] = metric_sol


        #figure(11)
        #plot(metric_sol['r'], zec_check(metric_sol, V_args, f_args))

    #lambdas = [252.0, 121.0**3.0, 972.0, 77.0**3.0] #T, s, mu, rho
    # Lambda_no = 831.0
    # kappa_5_no = 12.5
    # lambdas_no = [Lambda_no, kappa_5_no*Lambda_no**3.0, Lambda_no, kappa_5_no*Lambda_no**3.0]

    TD_T_axis_unscaled = TD_T_axis                      # unphysical solution in BH units
    TD_T_axis = TD_scale_isen(TD_T_axis, lambdas)[0]    # physical solution in MeV

    f_raster = rasterize(f, phi0_raster, *f_args)[1]

    return [TD_T_axis, phi0_raster, chiT2_raster, integral_raster, f_raster, f_args, metric_sols_list, TD_T_axis_unscaled]


def chi2_sT3(p, data):                          # (p, *args) for fmin_l_bfgs_b # (p, data) for amoeba
    """input:   p = [lambda_s, lambda_T, gamma, b, c4, c6],
                data = (phi0_min, phi0_max, phi0_pts, f_args)
      output:   -(chi squared) for s/T^3 compared to lattice data
    """
    lambda_s, lambda_T, gamma, b, c4, c6 = p
    phi0_min, phi0_max, phi0_pts, f_args = data # args
    print '\n\n\n\nlambda_s^1/3, lambda_T, gamma, b, c4, c6 = ', lambda_s**(1.0/3.0), lambda_T, gamma, b, c4, c6

    #+++++++++++++++++++++++ Thermo for mu=0:
    V_args = ['V_VI', np.array([gamma, b, c4, c6])]
    TDTA_mu0 = TD_calc_mu0(phi0_min, phi0_max, phi0_pts, V_args, f_args)
    if TDTA_mu0 == 0:
        return -100.0       # artificial value for bad BH solution
    TD_unscaled = TDTA_mu0[7]
    s_array = TD_unscaled[:,1]
    T_array = TD_unscaled[:,0]

    T_seq, s_seq = zip(*sorted(zip(T_array, s_array)))
    T_array = np.array(T_seq)
    s_array = np.array(s_seq)
    sT3_BH = lambda_s*s_array/(lambda_T*T_array)**3.0
    T_eval = lambda_T*T_array

    sT3_theo = splev(lat['T'], splrep(T_eval, sT3_BH))

    chi2 = np.sum( ( (sT3_theo-lat['sT3']) )**2.0 ) #/ (np.float(len(lat['T']))-2.0-1.0)
    print 'current chi^2 in V-fit: ', chi2

    return -chi2                                # ATTENTION:    -chi2 is returned for maximization!


def chi2_suscep(p, data):
    """input:   p = [args,...]
                data = (phi0_min, phi0_max, phi0_pts, V_args, lambda_T, f_type)
      output:   -(chi squared) for quark susceptibility chi2hat compared to lattice data
    """
    phi0_min, phi0_max, phi0_pts, V_args, lambda_T, f_type = data

    if f_type == 'f_no':
        lambda_rhomu, nrm, scl, shft, n2, efac = p
        print '\n\n\n\nlambda_rhomu, nrm, scl, shft, n2, efac = ', lambda_rhomu, nrm, scl, shft, n2, efac
        f_args = [f_type, np.array([nrm, scl, shft, n2, efac])]
    elif f_type == 'f_tanh':
        lambda_rhomu, const, nrm, scl, shft = p
        print '\n\n\n\nlambda_rhomu, const, nrm, scl, shft = ', lambda_rhomu, const, nrm, scl, shft
        f_args = [f_type, np.array([const, nrm, scl, shft])]

    #+++++++++++++++++++++++ Thermo for mu=0:
    TDTA_mu0 = TD_calc_mu0(phi0_min, phi0_max, phi0_pts, V_args, f_args)
    if TDTA_mu0 == 0:
        return -100.0       # artificial value for bad BH solution
    TD_unscaled = TDTA_mu0[7]
    T_array = TD_unscaled[:,0]
    chiT2_raster = TDTA_mu0[2]

    T_seq, chi2hat_seq = zip(*sorted(zip(T_array, chiT2_raster)))
    T_array = np.array(T_seq)
    chi2hat_array = np.array(chi2hat_seq)
    T_eval = lambda_T*T_array

    chi2hat_theo = splev(chi2_lat[:,0], splrep(T_eval, np.array(lambda_rhomu)/lambda_T**2.0*chi2hat_array)) # corrected missing lambda_T^2

    #plot(chi2_lat[:,0], chi2hat_theo)
    #plot(chi2_lat[:,0], chi2_lat[:,1], ls='', marker='s')
    #show()

    chi2 = np.sum( (chi2hat_theo-chi2_lat[:,1])**2 ) #/ np.float(len(chi2_lat[:,0]))
    print 'current chi^2 in f-fit: ', chi2

    return -chi2                                # ATTENTION:    -chi2 is returned for maximization!


def get_lambda_and_mu0sol(V_args, f_args, fit_V, fit_f, f_type):
    """
    Calculates the conversion factors lambda for a solution in physical units (MeV) and TD for mu=0
    Parameters for Potential V(phi) and Coupling f(phi) are fitted
    """

    # Fit sT3 to the lattice data at mu=0:
    #-------------------------------------
    if fit_V == 1:
        x0 =   [404**3.0, 900.0, 0.6, 0.7, -0.1, 0.0034]                           # initial values for lambda_s, lambda_T, gamma, b, c4, c6
        scale = [50**3.0, 200.0, 0.1, 0.1,  0.2, 0.01]
        #fit_sT3 = fmin_l_bfgs_b(chi2_sT3, x0, approx_grad=1, args=(phi0_min, phi0_max, phi0_pts, f_args),
                                #bounds=[(350**3.0,450**3.0),(850,950.0),(0.5,0.7),(0.5,0.9),(-1.0,1.0),(-0.2,0.2)], maxiter=400)
        fit_sT3 = amoeba(x0, scale, chi2_sT3, data=(phi0_min, phi0_max, phi0_pts, f_args))
        lambda_s = fit_sT3[0][0]
        lambda_T = fit_sT3[0][1]
        V_args_fitted = ['V_VI', np.array([fit_sT3[0][2], fit_sT3[0][3], fit_sT3[0][4], fit_sT3[0][5]])]
    else:
        lambda_s = lambdas[1] # global
        lambda_T = lambdas[0]
        V_args_fitted = V_args

    # Fit chi2T2 to the lattice data at mu=0:
    #----------------------------------------
    if fit_f == 1:
        if f_type == 'f_no':
            x0 =   [70000.0, 1.0/3.0*np.cosh(0.69), 1.2, 0.69/1.2, 2.0/3.0, -100.0]   # initial values for lambda_rhomu, nrm, scl, shft, n2, efac
            scale = [1000.0, 0.2                  , 0.2, 0.2     , 0.2    , 20.0]
            fit_chi2hat = amoeba(x0, scale, chi2_suscep,ftolerance=1.e-8, xtolerance=1.e-5, data=(phi0_min, phi0_max, phi0_pts, V_args_fitted, lambda_T, f_type))
            lambda_rhomu = fit_chi2hat[0][0]                # = lambda_rho/lambda_mu
            f_args_fitted = [f_type, np.array([fit_chi2hat[0][1], fit_chi2hat[0][2], fit_chi2hat[0][3], fit_chi2hat[0][4], fit_chi2hat[0][5]])]

        elif f_type == 'f_tanh':
            x0 =    [116.25 , 116.79, -116.03, 1.13, 2.35]   # initial values for lambda_rhomu, const, nrm, scl, shft
            scale = [10.0, 1.0, 1.0, 1.0, 1.0]
            fit_chi2hat = amoeba(x0, scale, chi2_suscep,ftolerance=1.e-4, xtolerance=1.e-4, data=(phi0_min, phi0_max, phi0_pts, V_args_fitted, lambda_T, f_type))
            lambda_rhomu = fit_chi2hat[0][0]
            f_args_fitted = [f_type, np.array([fit_chi2hat[0][1], fit_chi2hat[0][2], fit_chi2hat[0][3], fit_chi2hat[0][4]])]
        print 'lambda_rhomu =', '\t\t\t', lambda_rhomu, 'chi2: ', -fit_chi2hat[1]
    else:
        lambda_rhomu = lambdas[3]/lambdas[2]
        f_args_fitted = f_args

    # Thermo for mu=0 with new fitted values:
    #----------------------------------------
    TDTA_mu0 = TD_calc_mu0(phi0_min, phi0_max, phi0_pts, V_args_fitted, f_args_fitted)
    TD_unscaled = TDTA_mu0[7]
    chiT2_raster = TDTA_mu0[2]

    if not (fit_V == 0 and fit_f == 0):
        lambda_mu = np.sqrt( (lambda_s*splev(np.array(chi2_lat[:,0][-1]),
                                             splrep(lambda_T*TD_unscaled[:,0][::-1], chiT2_raster[::-1]))) / (lambda_T*1.0/3.0) )   # eqn (63)
        lambda_rho = lambda_rhomu*lambda_mu
    else:
        lambda_mu = lambdas[2]
        lambda_rho = lambdas[3]

    print 40*'='
    print '\nConversion Factors:'
    print 'lambda_s^1/3, lambda_T =', '\t', lambda_s**(1.0/3.0), lambda_T
    print 'lambda_mu, lambda_rho^1/3 =', '\t', lambda_mu, lambda_rho**(1.0/3.0)
    print 'lambda_rho/lambda_mu =', '\t', lambda_rhomu
    print '\nPotential Parameters:\n', V_args_fitted
    if fit_V==1:
        print 'chi^2 fit_sT3: ', -fit_sT3[1]
    print '\nCoupling Parameters:\n', f_args_fitted
    if fit_f==1:
        print 'chi^2 fit_chi2hat: ', -fit_chi2hat[1], '\n'
    print 40*'='


    return [TDTA_mu0, lambda_T, lambda_s, lambda_mu, lambda_rho, V_args_fitted, f_args_fitted]

################################################################################################################################################################



# Attention: global variables
model_type = 'no'
ftype = 'f_tanh'                    # chosen type for fit procedure
Vargs_initial = args_dic['V'][model_type]
fargs_initial = [ftype, np.array([116.79, -116.03, 1.13, 2.35])] # args_dic['f'][model_type] #
lambdas = lambdas_dic[model_type]   # <-- ATTENTION: Update these values in 'args_and_lambds.py' after fit (for other program files)

# Choose which fct-parameters to fit:
perform_V_fit = 1   # lambda_s, _T are also fitted
perform_f_fit = 1   # lambda_mu, _rho are fitted if any is set to 1



# fargs_initial = ['f_tanh', np.array([0.3886, 0.8198, -0.18438, 0.9171])]
# Lambda_no = 831.0
# kappa_5_no = 12.5
# chi2scale = 0.86805612028/12.5
#x0 = np.array(, 0.0173805]) #fit with V_G log chiq = 8.4909
#fargs_initial = ['f_tanh', np.array([1.962771, 0.019551, -3.327849, -0.180423])]
#fargs_initial = ['f_tanh', np.array([2.65172302, 2.24584626, -1.5673003, 3.94860272])]
#fargs_initial = ['f_tanh', np.array([19.94461, -29.09230, -0.013829, 44.57578])]
#lambdas[3] = 0.0173805*lambdas[3]


## Gubser:
if model_type == 'G':
    r_mid = 22.0

    phi0_min = 1.0
    phi0_max = 15.0

    Phi1r_max = 0.9

## Noronha:
if model_type == 'no':
    r_mid = 22.0

    #phi0_min = 0.52
    phi0_min = 0.01
    phi0_max = 6.0

    Phi1r_max = 0.99
##########################
eps = 1e-6
kappa_5 = 1.0
phi0_pts = 61
Phi1_pts = 41




#+++++++++++++++++++++++ lattice data:
lat = pickle.load(open('QG_latticedata_WuB.p','rb'))
file.close(open('QG_latticedata_WuB.p'))
chi2_lat = pickle.load(open('chi2_wubp.p', "rb"))
file.close(open('chi2_wubp.p'))
#+++++++++++++++++++++++



#print fargs_initial
phi0_raster0 = np.linspace(0, 500, 200)
f_raster0 = rasterize(f, phi0_raster0, *fargs_initial)[1]
#figure(1)
#plot(phi0_raster0, f_raster0)
#xlabel(r'$\phi$')
#ylabel(r'$f(\phi)$')
#show()



#+++++++++++++++++++++++ Thermo for mu=0:
GetLambdas = get_lambda_and_mu0sol(Vargs_initial, fargs_initial, perform_V_fit, perform_f_fit, ftype)
TDTA       = GetLambdas[0]
lambda_T   = GetLambdas[1]
lambda_s   = GetLambdas[2]
lambda_mu  = GetLambdas[3]
lambda_rho = GetLambdas[4]
Vargs      = GetLambdas[5]
fargs      = GetLambdas[6]
TD = TDTA[0]                    # in physical units from args file
TD_unscaled = TDTA[7]
phi0_raster = TDTA[1]
chiT2_raster = TDTA[2]
chiT2_integral_raster = TDTA[3]
f_raster = TDTA[4]
metric_sols_list = TDTA[6]



#+++++++++++++++++++++++ Save pickle:
fname = 'TDTA_'+model_type+'.p'
pickle.dump(TDTA[:-1], open(fname, "wb"))
file.close(open(fname))
# fname = 'metric_data_'+model_type+'.p'
# pickle.dump(metric_sols_list, open(fname, "wb"))
# file.close(open(fname))




############## s/T^3:
figure(1)
errorbar(lat['T'],lat['sT3'],lat['dsT3'], ls='', marker='s', ms=MS, mew=MEW, mec='k', c='k', label = 'WuBp data')
plot(lambda_T*TD_unscaled[:, 0], lambda_s*TD_unscaled[:, 1]/(lambda_T*TD_unscaled[:, 0])**3.0, c = 'b', label = model_type)
#plot(TD[:, 0], TD[:, 1]/TD[:, 0]**3.0, c = 'b', label = model_type)
axis([50, 800, 0, 20])
legend(frameon = 0, loc = 'lower right', numpoints=3)
xlabel(r'$T\, [MeV] $')
ylabel(r'$s(T, 0)/T^{\,3}$')
nice_ticks()
savefig('pdfs/sT3_mu0_'+model_type+'.pdf')
# s(T):
figure(2)
loglog(TD[:, 0], TD[:, 1])



############## chi_2:
print 'chiT2_raster: ', chiT2_raster
figure(3)
errorbar(chi2_lat[:,0], chi2_lat[:,1],chi2_lat[:,2], ls='', marker='s', ms=MS, mew=MEW, mec='k', c='k', label = 'WuBp data')
plot(lambda_T*TD_unscaled[:, 0], lambda_rho/(lambda_mu*lambda_T**2.0)*chiT2_raster, c = 'b', label = model_type)
#plot(TD[:, 0], chiT2_raster*lambdas[3]/(lambdas[2])/lambdas[0]**2.0)
axis([50, 800, 0, 0.4])
legend(frameon = 0, loc = 'lower right', numpoints=3)
xlabel(r'$T\, [MeV] $')
ylabel(r'$\chi_2(T, 0)/T^{\,2}$')
nice_ticks()
savefig('pdfs/chiT2_mu0_'+model_type+'.pdf')



#+++++++++++++++++++++ p-Calculation:
Phiphi_T0 = np.vstack((np.zeros(len(phi0_raster)), phi0_raster)) # for p_calcer_4
p_r = p_calc_line([TD[0,0],0],[TD[-1,0],0], TD, Phiphi_T0)
p_raster = p_r[0]
print 'p(T, mu = 0) =', p_raster

############## p/T^4:
figure(4)
errorbar(lat['T'],lat['pT4'],lat['dpT4'], ls='', marker='s', ms=MS, mew=MEW, mec='k', c='k', label = 'WuBp data')
plot(TD[:,0], p_raster/TD[:,0]**4.0, c = 'b', label = model_type)
axis([100, 550, 0, 4])
legend(frameon = 0, loc = 'lower right', numpoints=3)
xlabel(r'$T \, [MeV\,]$')
ylabel(r'$p(T, \mu = 0)/T^4$')
nice_ticks()
savefig('pdfs/pT4_mu0_'+model_type+'.pdf')



############## phi - dependence:
f_raster = rasterize(f, phi0_raster, *fargs)[1]
#figure(5)
#plot(TD[:,0], chiT2_integral_raster, lw = 2)
#figure(6)
#plot(TD[:,0], 1.0/chiT2_integral_raster, lw = 2)

figure(10)
plot(phi0_raster, TD[:,1]/TD[:,0]**3.0, label = r'$s/T^{\,3}$')
plot(phi0_raster, chiT2_raster, label = r'$\chi_2/T^{\,2}$')
plot(phi0_raster, 1.0/chiT2_integral_raster, label = r'$1/int$')
plot(phi0_raster, f_raster, label = r'$f(\phi)$')
plot(phi0_raster, 1.0/f_raster, label = r'$1/f(\phi)$')
plot(phi0_raster, 0.1*TD[:,0], label = r'$0.1T [MeV]$')
xlabel(r'$\phi_0$')
legend()

#figure(11)
#plot(phi0_raster, chiT2_integral_raster)
#plot(phi0_raster, f_raster)



############## vsq:
vsq_raster = vsq(TD, 'T')
figure(7)
errorbar(lat['T'],lat['cs2'],lat['dcs2'], ls='', marker='s', ms=MS, mew=MEW, mec='k', c='k', label = 'WuBp data')
plot(TD[:,0], vsq_raster, c = 'b', label = model_type)
axis([100, 550, 0, 0.4])
legend(frameon = 0, loc = 'lower right', numpoints=3)
xlabel(r'$T\, [MeV] $')
ylabel(r'$v_s^2$')
nice_ticks()
savefig('pdfs/vsq_mu0_'+model_type+'.pdf')



############## IT4:
print 'T_raster: ', TD[:,0]
IT4_raster = e_and_I(TD, p_raster[::-1])[1] / TD[:,0]**4.0 # (TD[:,1]*TD[:,0] - 4.0*p_raster[::-1])/TD[:,0]**4.0 # p_raster neeeds to be reversed - why?
print '\nI: ', IT4_raster
figure(8)
errorbar(lat['T'],lat['IT4'],lat['dIT4'], ls='', marker='s', ms=MS, mew=MEW, mec='k', c='k', label = 'WuBp data')
plot(TD[:,0], IT4_raster, c = 'b', label = model_type)
axis([100, 550, 0, 5])
legend(frameon = 0, loc = 'upper right', numpoints=3)
xlabel(r'$T\, [MeV] $')
ylabel(r'$I(T, \mu = 0)/T^4$')
nice_ticks()
savefig('pdfs/IT4_mu0_'+model_type+'.pdf')




#+++++++++++++++++++++ TD grids:
print '\ncurrent: backbone_g_5_mu0calc2.py --> with V-fit'
print '\n *** start calculating TD grids...'
nice_ticks()
show()

#t1 = time()

fname = 'TD_gr_'+model_type+'_wmu0.p'
TD_gr = TD_calc(phi0_min, phi0_max, phi0_pts, 0.0, Phi1_pts, Vargs, fargs, r_mid, eps, kappa_5)
pickle.dump(TD_gr, open(fname, "wb"))
file.close(open(fname))



fname = 'TD_gr_'+model_type+'.p'
TD_gr = TD_calc(phi0_min, phi0_max, phi0_pts, 0.02, Phi1_pts, Vargs, fargs, r_mid, eps, kappa_5)
pickle.dump(TD_gr, open(fname, "wb"))
file.close(open(fname))


#lambdas = [252.0, 121.0**3.0, 972.0, 77.0**3.0] #T, s, mu, rho
#epb_lamb = lambdas[1]/lambdas[3]
#
#phi0_pts = 80
#epbs = [6, 10, 20, 40]
#isentropes_dic = {}
#for epb in epbs:
#    phi0_min = 1.0
#    phi0_max = 15.0
#    phi0_bounds_new = phi0_range_isentrope(epb/epb_lamb, phi0_min, phi0_max, 400.0, Vargs, fargs)
#    print phi0_bounds_new
#    phi0_min = phi0_bounds_new[0]
#    phi0_max = phi0_bounds_new[1]
#    isen = isentrope_calc(epb/epb_lamb, phi0_min, phi0_max, phi0_pts, Vargs, fargs, r_mid, eps, kappa_5)
#    isentropes_dic.update({epb:isen})
#
#fname = 'TD_gr_G_isen2.p'
#pickle.dump(isentropes_dic, open(fname, "wb"))
#file.close(open(fname))

#############################################


print 'done: backbone_g_5_mu0calc2.py'