import numpy as np
from scipy.integrate import quad, odeint
from scipy.interpolate import splrep, splev
from amoba import amoeba

from Vtypes import Vs, dVs, dlogVs
from ftypes import fs, dfs
from H_expansions_G import get_horizon_expansions
from args_and_lambds import args_dic, lambdas_dic

from rasterizer import rasterize
from time import time
import pickle

#from pylab import figure, plot, legend, show, semilogy, loglog, axis, subplot, xlabel
#from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FixedLocator, LinearLocator

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
    print 'A_1 =', A_1, 'eps =', eps, 'r_mid =', r_mid
    #print 'ICs_h =', ICs_h
    
    r_raster = np.hstack((np.linspace(eps, r_mid/20.0, 2000), np.linspace(r_mid/20.0, r_mid, 2001)[1:]))
    #r_raster = np.hstack((np.linspace(eps**0.5, (r_mid/40.0)**0.5, 4000)**2.0, np.linspace(r_mid/40.0, r_mid,
                                                                                           #2001)[1:]))
    solution_1_wmsg = odeint(EFEs_G_full, ICs_h, r_raster, args = (V_args, f_args), atol = 1e-50, rtol = r_tol, full_output = True, mxstep = mxstp)
    #print solution_1_wmsg[1]['message'] 
    if solution_1_wmsg[1]['message'] != 'Integration successful.':
        raise ValueError
    else:
        solution_1 = solution_1_wmsg[0]
        
    sol1 = {'r':r_raster, 'A':solution_1[:, 0], 'dA':solution_1[:, 1], 'phi':solution_1[:, 2], 'dphi':solution_1[:, 3], 'h':solution_1[:, 4], 'dh':solution_1[:, 5], 'Phi':solution_1[:, 6], 'dPhi':solution_1[:, 7]}
    #print solution_1[:, 2][::100]
    if np.any(sol1['phi'] < 0):
        print 'negative phi, discarding solution...'
        raise ValueError
    
    if Phi_1 == 0:
        r_match = np.compress(sol1['dh']/sol1['h'] > 1e-30, r_raster)[-1]
        r_ind_match = len(np.compress(r_raster <= r_match, r_raster)) - 1
        print 'switching to 0T eqns at', r_match
    else:
        r_match = np.compress(sol1['dPhi']/sol1['Phi'] > 1e-30, r_raster)[-1]
        r_ind_match = len(np.compress(r_raster <= r_match, r_raster)) - 1
        print 'switching to 0T eqns at', r_match
    
    print 'r mid check:', r_mid, r_match, r_mid/40.0
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
    print 'r_inf =', r_inf
            
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
    rho = L*Q_G/(4.0*np.pi)*s
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
        Phi1r_raster = np.linspace(0.0, 0.9, Phi1_pts)
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
    if Phi_1r > 0:
        Phi_1_max = np.sqrt(-2.0*V(phi_0, *V_args)/f(phi_0, *f_args))
        Phi_1 = Phi_1r*Phi_1_max
        Q_G = f(phi_0, *f_args)*Phi_1
    elif Phi_1r == 0:
        Phi_1 = 0
        Q_G = 0

    Delta = Delta_from_Vargs(V_args)
    a_tol = 1e-80 #10.0**( - (4.0*(4.0 - Delta)*r_inf) - 16)
    eps = 1e-6
    #print 'Delta =', Delta, 'a_tol =', a_tol
    
    metric_sol = EFEs_solver_pw(phi_0, Phi_1, eps, r_mid, V_args, f_args, a_tol, r__tol, Delta)
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
    #print 'chi2T2 =', chi2T2

    if not return_integral:
        return chi2T2
    else:
        return [chi2T2, integral]

from fmg_TDprocess import TD_scale_isen
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
        print i, phi_0, msTD[0]#, TD_T_axis[i]
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

    #TD_T_axis = TD_scale_isen(TD_T_axis, lambdas)[0]
    # figure(1)
    # plot(TD_T_axis[:, 0], TD_T_axis[:, 1]/TD_T_axis[:, 0]**3.0)
    # figure(2)
    # loglog(TD_T_axis[:, 0], TD_T_axis[:, 1])
    # figure(3)
    # plot(TD_T_axis[:, 0], chiT2_raster*lambdas[3]/(lambdas[2])/lambdas[0]**2.0)
    #show()

    return [TD_T_axis, phi0_raster, chiT2_raster, integral_raster, metric_sols_list]


# model_type = 'G'
# V_args = args_dic['V'][model_type]
# f_args = args_dic['f'][model_type]
# lambdas = lambdas_dic[model_type]
# eps = 1e-6
# kappa_5 = 1.0
# phi0_pts = 81
# Phi1_pts = 41
# #############################################
# ## Gubser:
# if model_type == 'G':
#     r_mid = 22.0
#
#     phi0_min = 1.0
#     phi0_max = 15.0
#
#     Phi1r_max = 0.9
# #############################################
# ## Noronha:
# if model_type == 'no':
#     r_mid = 22.0
#
#     #phi0_min = 0.52
#     phi0_min = 0.1
#     phi0_max = 7.2
#
#     Phi1r_max = 0.5
# #############################################
# TDTA = TD_calc_mu0(phi0_min, phi0_max, phi0_pts, V_args, f_args)
# TD = TDTA[0]
# phi0_raster = TDTA[1]
# chiT2_raster = TDTA[2]
# chiT2_integral_raster = TDTA[3]
# metric_sols_list = TDTA[4]
#
# # fname = 'TDTA_'+model_type+'.p'
# # pickle.dump(TDTA, open(fname, "wb"))
# # file.close(open(fname))
# # fname = 'metric_data_'+model_type+'.p'
# # pickle.dump(metric_sols_list, open(fname, "wb"))
# # file.close(open(fname))
#
# from p_calcer2 import p_calc_line
#
# Phiphi_T0 = np.vstack((phi0_raster, np.zeros(len(phi0_raster))))
# p_r = p_calc_line([TD[0,0],0],[TD[-1,0],0], TD, Phiphi_T0)
#
# p_raster = p_r[0]
# print 'p(T, mu = 0) =', p_raster
#
# figure(3)
# axis([0, 1000, 0, 0.4])
#
# figure(4)
# plot(TD[:,0], p_raster/TD[:,0]**4.0, lw = 2)
# axis([100, 420, 0, 4])
# ax = subplot(111)
# ax.xaxis.set_major_locator(MultipleLocator(50))
# ax.xaxis.set_minor_locator(MultipleLocator(10))
#
# ax.yaxis.set_major_locator(MultipleLocator(1))
# ax.yaxis.set_minor_locator(MultipleLocator(0.2))
#
#
# f_raster = rasterize(f, phi0_raster, *f_args)[1]
# figure(5)
# plot(TD[:,0], chiT2_integral_raster, lw = 2)
# figure(6)
# plot(TD[:,0], 1.0/chiT2_integral_raster, lw = 2)
#
# figure(10)
# plot(phi0_raster, TD[:,1]/TD[:,0]**3.0, label = r'$s/T^{\,3}$')
# plot(phi0_raster, chiT2_raster, label = r'$\chi_2/T^{\,2}$')
# plot(phi0_raster, 1.0/chiT2_integral_raster, label = r'$1/int$')
# plot(phi0_raster, f_raster, label = r'$f(\phi)$')
# plot(phi0_raster, 1.0/f_raster, label = r'$1/f(\phi)$')
# plot(phi0_raster, 0.1*TD[:,0], label = r'$0.1T [MeV]$')
# xlabel(r'$\phi_0$')
# legend()
#
# figure(11)
# plot(phi0_raster, chiT2_integral_raster)
# plot(phi0_raster, f_raster)
#
# show()
#fname = 'TD_gr_G_wmu0.p'
#t1 = time()
#TD_gr = TD_calc(1.0, 15.0, phi0_pts, 0.02, Phi1_pts, V_args, f_args, r_mid, eps, kappa_5)
#pickle.dump(TD_gr, open(fname, "wb"))
#file.close(open(fname))
#
#lambdas = [252.0, 121.0**3.0, 972.0, 77.0**3.0] #T, s, mu, rho
#epb_lamb = lambdas[1]/lambdas[3]
#
#phi0_pts = 80
#epbs = [6, 10, 20, 40]
#isentropes_dic = {}
#for epb in epbs:
#    phi0_min = 1.0
#    phi0_max = 15.0
#    phi0_bounds_new = phi0_range_isentrope(epb/epb_lamb, phi0_min, phi0_max, 400.0, V_args, f_args)
#    print phi0_bounds_new
#    phi0_min = phi0_bounds_new[0]
#    phi0_max = phi0_bounds_new[1]
#    isen = isentrope_calc(epb/epb_lamb, phi0_min, phi0_max, phi0_pts, V_args, f_args, r_mid, eps, kappa_5)
#    isentropes_dic.update({epb:isen})
#
#fname = 'TD_gr_G_isen2.p'
#pickle.dump(isentropes_dic, open(fname, "wb"))
#file.close(open(fname))

#############################################
