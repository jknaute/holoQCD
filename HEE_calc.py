""" HEE_calc.py

Calculation of the holgraphic entanglement entropy (HEE) in the holographic QCD
model of Knaute et al. (1702.06731)
and plotting of the resulting phase diagram (PD)
"""

import numpy as np
import numpy.ma as ma
import math
import pickle
from pylab import *
from scipy.interpolate import splrep, splev, splint
from scipy.integrate import quad
#from mpmath import mp

from args_and_lambds import args_dic, lambdas_dic
from fmg_TDprocess import TD_scale, TD_scale_isen
from amoba import amoeba
from nice_tcks import nice_ticks

from HEE_addendum import pressure_like, nearest_neighbor_sort, get_phases, mask_grid, critical_behavior_hee, critical_behavior_TD


linew = 3
rc('font', size = 20) #fontsize of axis labels (numbers)
rc('axes', labelsize = 22, lw = linew) #fontsize of axis labels (symbols)
rc('lines', mew = 2, lw = linew, markeredgewidth = 2)
rc('patch', ec = 'k')
rc('xtick.major', pad = 7)
rc('ytick.major', pad = 7)
rcParams['mathtext.fontset'] = 'cm'
rcParams['mathtext.rm'] = 'serif'
rcParams['figure.figsize'] = [8.0, 6.0]

MS = 5              # marker size
MEW = 1             # marker edge width
lfs = 20            # legend font size
ff = 1.7            # for figsize
lw_for_pd = 10
CEP_dotsize = 35**2
clabel_fs = 18      # fontsize
set_labels_manually = 0
if set_labels_manually:
    InlineSpace = 12    # for clabels in PD
else:
    InlineSpace = 5


calculate_hee_grid = 0  # numerical calculation of hee   <==================   CHOOSE
save_matrix = 0         # save pickles w/ hee matrix

plot_hee_grid = 0       # plot previously calculated hee PD
save_figs = 0           # pdfs (also for levels)

calculate_hee_lvls = 0  # numerical calculation of hee for various mu levels
save_lvls = 0           # save pickles w/ hee on mu levels

plot_hee_lvls = 0       # plot previously calculated hee mu levels



#####################
model_type = 'VRY_4'
ftype = args_dic['ftype'][model_type]
V_args = args_dic['V'][model_type]
f_args = args_dic['f'][model_type]
lambdas = lambdas_dic[model_type]
#####################


###------- Read Files:
#fname = model_type+'/'+ftype+'/TDTA_'+model_type+'.p'
#TDTA = pickle.load(open(fname, "rb"))
#file.close(open(fname))

#### plotting metric functions for mu=0:
#figure(0)
#for i in range(0, len(TDTA[6])):
    ##print TDTA[6][i]['r'][0], TDTA[6][i]['r'][-1]
    #plot(TDTA[6][i]['r'], TDTA[6][i]['A'])
#show()


## preliminary global variable allocations:
#mp.dps = 100        # mpmath precision
l = 0.04            # width of entanglement region
r_m = 2.0           # large r cutoff
#r_m_ren = 2.0       # large r cutoff for renormalized definition
                    # (up to where difference def is numerically applicable)

T_CEP = 111.5
if model_type == 'VRY_2':
    mu_CEP = 987.5
elif model_type == 'VRY_4':
    mu_CEP = 611.5




def read_all_grid_files():
    global metric_gr, TD_grid, ij_tuples, phi0_pts, Phi1_pts

    fname = model_type+'/'+ftype+'/hee/TD_gr_'+model_type+'_wmu0_wmetric_part2.p'
    metric_gr = pickle.load(open(fname, "rb"))  # contains TD and (r, A, h) for every BH solution
    file.close(open(fname))

    TD_grid = metric_gr[0]
    TD_grid = TD_scale(TD_grid, lambdas)[0]     # scaled TD: [T, s, mu, n]
    ij_tuples = metric_gr[2]['metric_dic'].keys()
    phi0_pts = metric_gr[2]['phi0_pts']
    Phi1_pts = metric_gr[2]['Phi1_pts']
    ##-------


def read_TD_grid_file():
    global TD_grid

    if model_type == 'VRY_2':
        fname = model_type+'/'+ftype+'/hee/TD_gr_'+model_type+'_wmu0_wmetric_part2.p'
        metric_gr = pickle.load(open(fname, "rb"))  # contains TD and (r, A, h) for every BH solution
        file.close(open(fname))
    else:
        fname = model_type+'/'+ftype+'/hee/TD_gr_'+model_type+'_wmu0_forhee.p'
        metric_gr = pickle.load(open(fname, "rb"))  # contains TD
        file.close(open(fname))

    TD_grid = metric_gr[0]
    TD_grid = TD_scale(TD_grid, lambdas)[0]     # scaled TD: [T, s, mu, n]
    ##-------


def function_integrand(r_vals, tck):
    """spline for integration in:
    -> boundary condition (bc), and holographic entanglement entropy (hee)
    """
    return splev(r_vals, tck)


def chi2_length(p, data):
    """calculate l/2-int in bc for minimization in r_star calculation"""

    rstar = p[0]
    r_vals, A_vals, h_vals = data

    ### calculate integrand:
    Astar = splev(rstar, splrep(r_vals, A_vals))
    bc_integrand = np.array(( h_vals*(np.exp(8.0*A_vals - 6.0*Astar) - np.exp(2.0*A_vals)) )**(-1.0/2.0))

    ### masking:
    bc_integrand = ma.asarray(bc_integrand)
    r_vals = ma.asarray(r_vals)
    for k in range(0, len(bc_integrand)):
        if math.isnan(bc_integrand[k])==1:
            bc_integrand[k] = ma.masked
            r_vals[k] = ma.masked

    bc_integrand = bc_integrand[~bc_integrand.mask]     # choose valid entries (no 'nan's)
    r_vals = r_vals[~r_vals.mask]

    ### integration:
    bc_integrand_tck = splrep(r_vals, bc_integrand, k=1)
    bc_integral = quad(function_integrand, rstar, r_m, args=(bc_integrand_tck,), epsabs = 1e-50, epsrel = 1e-13, limit = 1000)[0]
    chi2 = (l/2.0 - bc_integral)**2.0

    return -chi2                                        # ATTENTION:    -chi2 is returned for maximization!


def hee_integration(r_array, A_array, h_array):
    """
    Calculation of S_HEE and S_HEE_ren
    """

    ### fit procedure to calculate r_star:
    x0 = [0.2]
    scale = [0.05]
    fit_rstar = amoeba(x0, scale, chi2_length, data=(r_array, A_array, h_array))
    r_star = fit_rstar[0][0]
    print 'r_star    = ', r_star, '\t', fit_rstar[2], -fit_rstar[1]


    ### calculate integrands:
    A_star = splev(r_star, splrep(r_array, A_array))
    hee_integrand = (np.exp(5.0*A_array - 3.0*A_star)) / (np.sqrt(h_array*(np.exp(6.0*A_array - 6.0*A_star) - 1.0)))
    hee_integrand_reg = (np.exp(5.0*A_array)) / (np.sqrt(h_array*(np.exp(6.0*A_array) - 1.0)))   # regularized definition
    hee_integrand_ren = np.log(hee_integrand / hee_integrand_reg)   # renormalized definition


    ### masking:
    r_array_ren = ma.asarray(r_array)                   # separate masking for ren def
    r_array = ma.asarray(r_array)

    hee_integrand = ma.asarray(hee_integrand)           # masking hee def
    for k in range(0, len(hee_integrand)):
        if math.isnan(hee_integrand[k])==1:
            hee_integrand[k] = ma.masked
            r_array[k] = ma.masked
    hee_integrand = hee_integrand[~hee_integrand.mask]  # choose valid entries (no 'nan's)
    r_array = r_array[~r_array.mask]

    hee_integrand_ren = ma.asarray(hee_integrand_ren)   # masking ren def
    for k in range(0, len(hee_integrand_ren)):
        if math.isnan(hee_integrand_ren[k])==1:
            hee_integrand_ren[k] = ma.masked
            r_array_ren[k] = ma.masked
    hee_integrand_ren = hee_integrand_ren[~hee_integrand_ren.mask]
    r_array_ren = r_array_ren[~r_array_ren.mask]

    for k in range(len(hee_integrand_ren)):
        if hee_integrand_ren[k+1] > hee_integrand_ren[k]:
            r_m_ren_ind = k
            break
    r_m_ren = r_array_ren[r_m_ren_ind]
    inds_ren = np.where(r_array_ren<r_m_ren)            # small r indices where ren def can be applied
    print 'r_m_ren   = ', r_m_ren


    ### final HEE integrals:
    hee_integrand_tck = splrep(r_array, hee_integrand)
    hee_integrand_ren_tck = splrep(r_array_ren[inds_ren], hee_integrand_ren[inds_ren])

    S_HEE = 0.5*quad(function_integrand, r_star, r_m, args=(hee_integrand_tck,), epsabs = 1e-9, epsrel = 1e-13, limit = 1000)[0]
    S_HEE_ren = 0.5*quad(function_integrand, r_star, r_m_ren, args=(hee_integrand_ren_tck,), epsabs = 1e-15, epsrel = 1e-13, limit = 1000)[0]

    return [S_HEE, S_HEE_ren]




##===============================   grid   ====================================

def calculate_HEE_grid():
    """
    Numerical Calculation of  HEE (grid)
    for every BH solution parameterized by (phi0, Phi1)
    """
    read_all_grid_files()

    T_list = []
    S_HEE_mu0_list = []
    S_HEE_mu0_ren_list = []
    S_HEE_matrix = np.zeros((phi0_pts, Phi1_pts))
    S_HEE_ren_matrix = np.zeros((phi0_pts, Phi1_pts))

    iterate = 0
    for ij in ij_tuples:    # loop over phi0, Phi1 values (indices):
        iterate = iterate + 1
        i, j = ij
        T = TD_grid[i,j,0]
        mu = TD_grid[i,j,2]
        print '\ni, j      = ', ij, '   ', iterate, '/', len(ij_tuples)
        print 'T, mu     = ', T, mu
        if math.isnan(T)==1 or math.isnan(mu)==1:       # bad BH solution
            print 'disregarded'
            continue

        r_vals = metric_gr[2]['metric_dic'][ij]['r']
        A_vals = metric_gr[2]['metric_dic'][ij]['A']
        h_vals = metric_gr[2]['metric_dic'][ij]['h']

        hee = hee_integration(r_vals, A_vals, h_vals)   # contains hee integral evaluations

        S_HEE = hee[0]
        S_HEE_ren = hee[1]
        S_HEE_matrix[i,j] = S_HEE
        S_HEE_ren_matrix[i,j] = S_HEE_ren
        print 'S_HEE     = ', S_HEE
        print 'S_HEE_ren = ', S_HEE_ren
        if j==0:
            T_list.append(T)
            S_HEE_mu0_list.append(S_HEE)
            S_HEE_mu0_ren_list.append(S_HEE_ren)


    ##+++++++++++ Saving
    print '\nS_HEE_matrix: \n', S_HEE_matrix
    print '\nS_HEE_ren_matrix: \n', S_HEE_ren_matrix
    if save_matrix:
        fname = model_type+'/'+ftype+'/hee/HEE_matrix.p'
        pickle.dump(S_HEE_matrix, open(fname, "wb"))
        file.close(open(fname))

        fname = model_type+'/'+ftype+'/hee/HEE_ren_matrix.p'
        pickle.dump(S_HEE_ren_matrix, open(fname, "wb"))
        file.close(open(fname))


    ##+++++++++++ Plotting
    if len(S_HEE_mu0_list)>0:
        figure(2)
        semilogy(np.array(T_list), np.array(S_HEE_mu0_list), ls='', marker='s')
        xlabel(r'$T \, [MeV\,]$')
        ylabel(r'$S_{HEE}$')

        figure(3)
        plot(np.array(T_list), np.array(S_HEE_mu0_ren_list), ls='', marker='s')
        xlabel(r'$T \, [MeV\,]$')
        ylabel(r'$S_{HEE}^{ren}$')
    ##-------



def plot_HEE_grid():
    """Plotting of HEE PD"""

    if calculate_hee_grid != 1:  # otherwise already read out
        read_TD_grid_file()

    rc('font', size = 30) #fontsize of axis labels (numbers)
    rc('axes', labelsize = 38, lw = 2) #fontsize of axis labels (symbols)
    rc('xtick.major', pad = 2)
    rc('ytick.major', pad = 7)

    if model_type == 'VRY_2':
        fname = model_type+'/'+ftype+'/hee/HEE_matrix_rm2_part2.p'
        S_HEE_matrix = pickle.load(open(fname, "rb"))
        file.close(open(fname))

        fname = model_type+'/'+ftype+'/hee/HEE_ren_matrix_rm2_part2.p'
        S_HEE_ren_matrix = pickle.load(open(fname, "rb"))
        file.close(open(fname))
    else:
        fname = model_type+'/'+ftype+'/hee/HEE_matrix_rm2.p'
        S_HEE_matrix = pickle.load(open(fname, "rb"))
        file.close(open(fname))

        fname = model_type+'/'+ftype+'/hee/HEE_ren_matrix_rm2.p'
        S_HEE_ren_matrix = pickle.load(open(fname, "rb"))
        file.close(open(fname))
    print '\nS_HEE_matrix: \n', S_HEE_matrix
    print '\nS_HEE_ren_matrix: \n', S_HEE_ren_matrix

    fname = model_type+'/'+ftype+'/hee/Tc_muT_hee.p'
    Tc_muT_hee = pickle.load(open(fname, "rb"))
    file.close(open(fname))


    #--- S_HEE grid:
    S_HEE_matrix_m = mask_grid(S_HEE_matrix, TD_grid, Tc_muT_hee, 'all')    # phase = 'htp' or 'ltp' or 'all'
    S_HEE_matrix_ltp = mask_grid(S_HEE_matrix, TD_grid, Tc_muT_hee, 'htp')
    S_HEE_matrix_htp = mask_grid(S_HEE_matrix, TD_grid, Tc_muT_hee, 'ltp')

    #hee_levels = np.linspace(np.log(S_HEE_matrix_m).min(), np.log(S_HEE_matrix_m).max(), 400)
    hee_levels = np.linspace(12, 40, 400)
    #div_lim = 0.5
    #hee_levels_2a = np.linspace(np.log(S_HEE_matrix_m).min(), div_lim*np.log(S_HEE_matrix_m).max(), 10)
    #hee_levels_2b = np.linspace(div_lim*np.log(S_HEE_matrix_m).max(), np.log(S_HEE_matrix_m).max(), 5)
    #hee_levels_2 = np.hstack((hee_levels_2a, hee_levels_2b))
    hee_levels_2 = np.array([14.0, 15.0, 16.0, 17.0, 18.0, 20.0, 22.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0, 38.0])
    hee_ticks = list(np.arange(0, np.amax(hee_levels)*1.001, np.amax(hee_levels)/10.0))

    hee_colors_2 = [None]*len(hee_levels_2)
    for i in range(0, len(hee_colors_2)):
        hee_colors_2[i] = cm.Greens( (hee_levels_2[i]-hee_levels_2.min())/(np.amax(hee_levels_2)-hee_levels_2.min()) )

    figure(4, figsize = (ff*8.0, ff*6.2))
    #contourf(TD_grid[:,:,2], TD_grid[:,:,0], np.log(S_HEE_matrix_ltp), levels = hee_levels, cmap = cm.jet, zorder=-5)
    contourf(TD_grid[:,:,2]/mu_CEP, TD_grid[:,:,0]/T_CEP, np.log(S_HEE_matrix_htp), levels = hee_levels, cmap = cm.jet, zorder=-4)
    contourf(TD_grid[:,:,2]/mu_CEP, TD_grid[:,:,0]/T_CEP, np.log(S_HEE_matrix_m), levels = hee_levels, cmap = cm.jet, zorder=-3)
    nice_ticks()
    if model_type == 'VRY_2':
        axis([0.0, 1450.0/mu_CEP, 50.0/T_CEP, 200.0/T_CEP])
    else:
        axis([0.0, 1.5, 0.5, 1.8])
        ax = subplot(111)
        ax.set_xticks(np.arange(0, 1.5, 0.2))
    colorbar(spacing = 'proportional', ticks = hee_ticks)

    hee_lvls = contour(TD_grid[:,:,2]/mu_CEP, TD_grid[:,:,0]/T_CEP, np.log(S_HEE_matrix_m), levels = hee_levels_2, colors = hee_colors_2, zorder=-2)
    clabel(hee_lvls, hee_levels_2, inline = 1, fontsize = clabel_fs, fmt = '%1.1f', manual=set_labels_manually, inline_spacing=InlineSpace)

    plot(Tc_muT_hee[0,:]/mu_CEP, Tc_muT_hee[1,:]/T_CEP, color = 'grey', lw=lw_for_pd, zorder=-1)
    scatter(1.0, 1.0, c='white', s=CEP_dotsize, alpha=1, zorder=1, edgecolor='white')

    xlabel(r'$\mu/\mu_{CEP}$')
    ylabel(r'$T/T_{CEP}$')
    title(r'$\ln \, S_{\mathrm{HEE}}^{reg}$', y=1.02)
    if save_figs:
        savefig(model_type+'/'+ftype+'/pdfs/HEE/S_HEE_reg_'+model_type+'_PD.pdf')


    #--- S_HEE_ren grid:
    S_HEE_ren_matrix_m = S_HEE_ren_matrix # mask_grid(S_HEE_ren_matrix, TD_grid, Tc_muT_hee, 'all')
    #S_HEE_ren_matrix_ltp = mask_grid(S_HEE_ren_matrix, TD_grid, Tc_muT_hee, 'htp')
    #S_HEE_ren_matrix_htp = mask_grid(S_HEE_ren_matrix, TD_grid, Tc_muT_hee, 'ltp')

    #hee_levels = np.linspace(S_HEE_ren_matrix_m.min(), S_HEE_ren_matrix_m.max(), 400)
    hee_levels = np.linspace(0, 0.02, 400)
    div_lim = 0.5
    hee_levels_2a = np.linspace(S_HEE_ren_matrix_m.min(), div_lim*S_HEE_ren_matrix_m.max(), 10)
    hee_levels_2b = np.linspace(div_lim*S_HEE_ren_matrix_m.max(), S_HEE_ren_matrix_m.max(), 5)
    hee_levels_2 = np.hstack((hee_levels_2a, hee_levels_2b))
    #hee_levels_2 = np.array([14.0, 15.0, 16.0, 17.0, 18.0, 20.0, 22.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0, 38.0])
    hee_ticks = list(np.arange(0, np.amax(hee_levels)*1.001, np.amax(hee_levels)/10.0))

    hee_colors_2 = [None]*len(hee_levels_2)
    for i in range(0, len(hee_colors_2)):
        hee_colors_2[i] = cm.Greens( (hee_levels_2[i]-hee_levels_2.min())/(np.amax(hee_levels_2)-hee_levels_2.min()) )

    figure(41, figsize = (ff*8.0, ff*6.2))
    #contourf(TD_grid[:,:,2], TD_grid[:,:,0], S_HEE_matrix_ltp, levels = hee_levels, cmap = cm.jet, zorder=-5)
    #contourf(TD_grid[:,:,2], TD_grid[:,:,0], S_HEE_matrix_htp, levels = hee_levels, cmap = cm.jet, zorder=-4)
    contourf(TD_grid[:,:,2], TD_grid[:,:,0], S_HEE_ren_matrix_m, levels = hee_levels, cmap = cm.jet, zorder=-3)
    nice_ticks()
    if model_type == 'VRY_2':
        axis([0.0, 1450.0/mu_CEP, 50.0/T_CEP, 200.0/T_CEP])
    else:
        axis([0.0, 1.5*mu_CEP, 0.5*T_CEP, 1.8*T_CEP])
    colorbar(spacing = 'proportional', ticks = hee_ticks)

    #hee_lvls = contour(TD_grid[:,:,2], TD_grid[:,:,0], S_HEE_ren_matrix_m, levels = hee_levels_2, colors = hee_colors_2, zorder=-2)
    #clabel(hee_lvls, hee_levels_2, inline = 1, fontsize = clabel_fs, fmt = '%1.1f', manual=set_labels_manually)

    #plot(Tc_muT_hee[0,:], Tc_muT_hee[1,:], color = 'grey', lw=lw_for_pd, zorder=-1)
    #scatter(mu_CEP, T_CEP, c='white', s=CEP_dotsize, alpha=1, zorder=1, edgecolor='white')

    ylabel(r'$T \, [MeV\,]$')
    xlabel(r'$\mu \, [MeV\,]$')
    title(r'$S_{\mathrm{HEE}}^{ren}$', y=1.02)
    if save_figs:
        savefig(model_type+'/'+ftype+'/pdfs/HEE/S_HEE_ren_'+model_type+'_PD.pdf')
    ##-------





##==============================   levels   ===================================

def calculate_HEE_lvls():
    """
    Numerical Calculation of  HEE (levels)
    along lines of constant mu for different T
    """
    HEE_mulvls = {}
    HEE_ren_mulvls = {}
    files = ['part5'] # ['part0', 'part13', 'part2', 'part4'] # one file would be too large

    for File in files:
        fname = model_type+'/'+ftype+'/hee/metric_dic_mulvls_'+File+'.p'
        metric_dic_mulvls = pickle.load(open(fname, "rb"))
        file.close(open(fname))

        mu_levels = np.sort(metric_dic_mulvls.keys())

        iterate = 0
        for mu in mu_levels:
            T_levels = np.sort(metric_dic_mulvls[mu].keys())
            T_list = []
            S_HEE_list = []
            S_HEE_ren_list = []

            for T in T_levels:
                iterate = iterate + 1
                print '\nmu, T     = ', mu, T, '   ', iterate, '/', len(mu_levels)*len(T_levels), '  in ', File
                r_vals = metric_dic_mulvls[mu][T]['r']
                A_vals = metric_dic_mulvls[mu][T]['A']
                h_vals = metric_dic_mulvls[mu][T]['h']

                hee = hee_integration(r_vals, A_vals, h_vals)   # contains hee integral evaluations

                S_HEE = hee[0]
                S_HEE_ren = hee[1]
                print 'S_HEE     = ', S_HEE
                print 'S_HEE_ren = ', S_HEE_ren
                T_list.append(T)
                S_HEE_list.append(S_HEE)
                S_HEE_ren_list.append(S_HEE_ren)

            ## update HEE_mulvls dictionary for current mu level:
            HEE_mulvls[mu] = {'T':np.array(T_list), 'S_HEE':np.array(S_HEE_list)}
            HEE_ren_mulvls[mu] = {'T':np.array(T_list), 'S_HEE_ren':np.array(S_HEE_ren_list)}


    ##+++++++++++ Saving
    print '\nHEE_mulvls: \n', np.sort(HEE_mulvls.keys())
    if save_lvls:
        fname = model_type+'/'+ftype+'/hee/HEE_mulvls.p'
        pickle.dump(HEE_mulvls, open(fname, "wb"))
        file.close(open(fname))

        fname = model_type+'/'+ftype+'/hee/HEE_ren_mulvls.p'
        pickle.dump(HEE_ren_mulvls, open(fname, "wb"))
        file.close(open(fname))
    ##-------



def plot_HEE_lvls():
    """
    Plotting HEE along mu levels
    and determination of phase contour
    """

    ## Files:
    fname = model_type+'/'+ftype+'/hee/HEE_mulvls_rm2.p'
    HEE_mulvls = pickle.load(open(fname, "rb"))
    file.close(open(fname))

    fname = model_type+'/'+ftype+'/hee/HEE_ren_mulvls_rm2.p'
    HEE_ren_mulvls = pickle.load(open(fname, "rb"))
    file.close(open(fname))

    if model_type == 'VRY_2':
        fname = model_type+'/'+ftype+'/phase_contour_and_spinodals_VRY_2.p'
        phase_contour_and_spinodals = pickle.load(open(fname, "rb"))
        file.close(open(fname))

        mu_levels = np.sort(HEE_mulvls.keys())
        print 'mu_levels: ', mu_levels
        mu_plotlist = [0.0, 400.0, 800.0, 1000.0, 1200.0, 1400.0]
        mu_plotlist2 = list(np.arange(985.0, 996, 1)) # for narrow excerpt

        Tc_muT = phase_contour_and_spinodals['phase_contour']['Tc_mu_T']    # FOPT curve from 'true' TD

        fname = model_type+'/'+ftype+'/hee/T_mu_contours_'+model_type+'_mupts_part5.p'   # normal TD on mulvl for comparison and critical scaling
        Tcont = pickle.load(open(fname, "rb"))
        file.close(open(fname))

        acc_cntrTD = Tcont['mu'][987.25]
        TD_slice_scaled = TD_scale_isen(acc_cntrTD[1], lambdas)[0]
    else:
        fname = model_type+'/'+ftype+'/Tc_muT.p'
        Tc_muT = pickle.load(open(fname, "rb")) # FOPT curve from 'true' TD
        file.close(open(fname))

        mu_levels = np.sort(HEE_mulvls.keys())
        print 'mu_levels: ', mu_levels
        mu_plotlist = list(np.array([0.0, 0.4, 0.8, 1.0, 1.2, 1.4])*mu_CEP)
        mu_plotlist1 = list(np.array([0.8, 1.0, 1.2])*mu_CEP)
        colors1 = ['C2', 'C3', 'C4']
        mu_plotlist2 = list(np.arange(605.0, 616, 1)) # for narrow excerpt

        fname = model_type+'/'+ftype+'/T_mu_contours_'+model_type+'_muCEP.p'   # normal TD on mulvl for comparison and critical scaling
        Tcont = pickle.load(open(fname, "rb"))
        file.close(open(fname))

        acc_cntrTD = Tcont['mu'][611.0]
        TD_slice_scaled = TD_scale_isen(acc_cntrTD[1], lambdas)[0]


    ## Variables:
    mu_c = []
    T_c = []
    htp_T_min = []  # spinodal pts
    ltp_T_max = []
    S_up = []       # 3 values of S_HEE in multivalued branches
    S_mid = []      # (for masking)
    S_down = []

    for mu in mu_levels:
        print 'mu = ', mu

        ## TD:
        T = HEE_mulvls[mu]['T']
        S_HEE = HEE_mulvls[mu]['S_HEE']
        S_HEE_ren = HEE_ren_mulvls[mu]['S_HEE_ren']

        if T[0] > T[-1]: # ordered by increasing T
            T = T[::-1]
            S_HEE = S_HEE[::-1]
            S_HEE_ren = S_HEE_ren[::-1]

        T, S_HEE_ln = nearest_neighbor_sort(T, np.log(S_HEE))   # S_HEE -> ln(S_HEE) from now on
        p_HEE = pressure_like(S_HEE_ln, T)                      # from:  dp_HEE = ln(S_HEE) dT

        ## Phase type:
        PhaseType = get_phases(p_HEE, T)
        print 'mu, PT: ', mu, '\t', PhaseType[0]
        print 'T_c, htp_T_min, ltp_T_max = ', PhaseType[1], '\t', PhaseType[2], '\t', PhaseType[3]
        if PhaseType[0]=='1st':
            ltp_max_ind = PhaseType[4]
            htp_min_ind = PhaseType[5]
            mu_c.append(mu)
            Tc = PhaseType[1]
            T_c.append(Tc)
            htp_T_min.append(PhaseType[2])
            ltp_T_max.append(PhaseType[3])
            S_up.append(float(splev(Tc, splrep(T[:ltp_max_ind], S_HEE_ln[:ltp_max_ind]))))
            S_down.append(float(splev(Tc, splrep(T[htp_min_ind:], S_HEE_ln[htp_min_ind:]))))
            if (htp_min_ind-ltp_max_ind) < 3:
                S_mid.append(float(splev(Tc, splrep(T[ltp_max_ind:htp_min_ind+1][::-1], S_HEE_ln[ltp_max_ind:htp_min_ind+1][::-1], k=1)))) # linear interpolation
            else:
                S_mid.append(float(splev(Tc, splrep(T[ltp_max_ind:htp_min_ind+1][::-1], S_HEE_ln[ltp_max_ind:htp_min_ind+1][::-1])))) # default 3rd degree interpolation
        else:
            mu_lvl_2nd = mu     # mu_lvl nearest to CEP before FOPT starts (is 2nd order PT)

        if mu in mu_plotlist:
            figure(5) # S_HEE
            plot(T/T_CEP, S_HEE_ln, label = r'%2.1f' %(mu/mu_CEP))
            figure(51) # S_HEE_ren
            plot(T/T_CEP, S_HEE_ren, label = r'$\mu = %d \, MeV$' %mu, ls='', marker='s', ms=2)
        if mu in mu_plotlist1:
            figure(7) # p_HEE
            plot(T, p_HEE, c=colors1[np.where(mu_plotlist1==mu)[0][0]], label = r'%2.1f' %(mu/mu_CEP))
        if mu in mu_plotlist2:
            figure(6) # S_HEE excerpt
            plot(T, S_HEE_ln, label = r'$\mu = %d \, MeV$' %mu)

    ## Critical Exponent:
    print 'mu_lvl_2nd = ', mu_lvl_2nd
    CritExp_hee = CritExp_TD = 0
    if model_type == 'VRY_2':
        CritExp_hee = critical_behavior_hee(HEE_mulvls[mu_lvl_2nd]['T'], np.log(HEE_mulvls[mu_lvl_2nd]['S_HEE']))
    else:
        CritExp_hee = critical_behavior_hee(HEE_mulvls[mu_lvl_2nd]['T'][::-1], np.log(HEE_mulvls[mu_lvl_2nd]['S_HEE'][::-1]))
    #CritExp_TD  = critical_behavior_TD(TD_slice_scaled[:,0], TD_slice_scaled[:,1])


    ## FOPTs and spinodals:
    Tc_muT_hee = np.vstack((mu_c, T_c, htp_T_min, ltp_T_max, S_up, S_mid, S_down))  # <== FOPT information
    if model_type == 'VRY_2':
        mu_help = np.linspace(950.0, 1050.0, 500)
    if model_type == 'VRY_4':
        mu_help = np.linspace(550.0, 650.0, 500)
    print '\nmu_lvl_2nd = ', mu_lvl_2nd
    print 'Tc_muT_hee: \n', Tc_muT_hee, len(mu_c)

    figure(8)
    plot(Tc_muT_hee[0,:]/mu_CEP, Tc_muT_hee[1, :]/T_CEP, color = 'grey', label='HEE')
    plot(Tc_muT[0,:]/mu_CEP, Tc_muT[1,:]/T_CEP, c = 'b', ls='--', label='Thermo')
    scatter(1.0, 1.0, c='red', s=20**2, alpha=1, edgecolor='red')
    if model_type == 'VRY_2':
        axis([950/mu_CEP, 1600/mu_CEP, 60/T_CEP, 115/T_CEP])
    if model_type == 'VRY_4':
        axis([0.95, 1.65, 0.5, 1.05])

    figure(9)
    plot(Tc_muT_hee[0,:], Tc_muT_hee[1,:], c = 'b', label='$T_c(\mu_c)$ hee')
    plot(Tc_muT_hee[0,:], Tc_muT_hee[2,:], c = 'r', label='htp spinodal')
    plot(Tc_muT_hee[0,:], Tc_muT_hee[3,:], c = 'g', label='ltp spinodal')
    plot(mu_help, splev(mu_help, splrep(Tc_muT_hee[0,:], Tc_muT_hee[1,:], k=1)), c = 'b', ls='--')
    plot(mu_help, splev(mu_help, splrep(Tc_muT_hee[0,:], Tc_muT_hee[2,:], k=1)), c = 'r', ls='--')
    plot(mu_help, splev(mu_help, splrep(Tc_muT_hee[0,:], Tc_muT_hee[3,:], k=1)), c = 'g', ls='--')

    for fig in [8, 9]: # FOPT, spinodals
        figure(fig)
        nice_ticks()
        if fig==8:
            legend(loc = 'best', numpoints=3, fontsize = lfs, frameon = 0, fancybox = 0, columnspacing = 1, title=r'$T_c(\mu_c):$')
            xlabel(r'$\mu/\mu_{CEP}$')
            ylabel(r'$T/T_{CEP}$')
        else:
            legend(loc='best')
            xlabel(r'$\mu \, [MeV\,]$')
            ylabel(r'$T \, [MeV\,]$')
        if save_figs:
            savefig(model_type+'/'+ftype+'/pdfs/HEE/FOPTs'+str(fig)+'.pdf')
    for fig in [5]:  # S_HEE_reg(T/T_CEP)
        figure(fig)
        xlabel(r'$T/T_{CEP}$')
        ylabel(r'$\ln \, S_{\mathrm{HEE}}^{reg}$')
        if model_type == 'VRY_4':
            axis([0, 5, 10, 40])
        nice_ticks()
        legend(loc = 'best', numpoints=3, fontsize = lfs, frameon = 0, fancybox = 0, columnspacing = 1, title=r'$\mu / \mu_{CEP}\, = $')
        if save_figs:
            savefig(model_type+'/'+ftype+'/pdfs/HEE/S_HEE_mulvls'+str(fig)+'.pdf')
    for fig in [51]:  # S_HEE_ren(T/T_CEP)
        figure(fig)
        xlabel(r'$T/T_{CEP}$')
        ylabel(r'$S_{\mathrm{HEE}}^{ren}$')
        if model_type == 'VRY_4':
            axis([0, 5, 0.006, 0.02])
        nice_ticks()
        #legend(loc = 'best', numpoints=3, fontsize = lfs, frameon = 0, fancybox = 0, columnspacing = 1, title=r'$\mu / \mu_{CEP}\, = $')
        if save_figs:
            savefig(model_type+'/'+ftype+'/pdfs/HEE/S_HEE_ren_mulvls'+str(fig)+'.pdf')
    for fig in [6]:  # S_HEE(T) excerpt
        figure(fig)
        xlabel(r'$T \, [MeV\,]$')
        ylabel(r'$\ln \, S_{\mathrm{HEE}}^{reg}$')
        nice_ticks()
        legend(loc = 'best', numpoints=3, fontsize = lfs, frameon = 0, fancybox = 0, columnspacing = 1)
        if save_figs:
            savefig(model_type+'/'+ftype+'/pdfs/HEE/S_HEE_mulvls'+str(fig)+'.pdf')
    for fig in [7]:     # p(T)
        figure(fig)
        xlabel(r'$T \, [MeV\,]$')
        ylabel(r'$p_{\mathrm{HEE}}$')
        axis([95, 115, 1750, 2300])
        nice_ticks()
        legend(loc = 'best', numpoints=3, fontsize = lfs, frameon = 0, fancybox = 0, columnspacing = 1, title=r'$\mu / \mu_{CEP}\, = $')
        if save_figs:
            savefig(model_type+'/'+ftype+'/pdfs/HEE/p_HEE_mulvls.pdf')

    for fig in [100]:     # ln(s) - ln(T)
        figure(fig)
        if CritExp_hee:
            xlabel(r'$\ln\vert \hat s-\hat s_{CEP} \vert$')
            ylabel(r'$\ln\vert T-T_{CEP} \vert$')
            nice_ticks()
            legend(loc = 'best', numpoints=3, fontsize = lfs, frameon = 0, fancybox = 0, columnspacing = 1, title='HEE')
            if save_figs:
                savefig(model_type+'/'+ftype+'/pdfs/HEE/linfit_HEE.pdf')
        if CritExp_TD:
            xlabel(r'$\ln\vert s-s_{CEP} \vert$')
            ylabel(r'$\ln\vert T-T_{CEP} \vert$')
            nice_ticks()
            dummy = matplotlib.patches.Rectangle((0,0), 1, 1, fill=False, edgecolor='none', visible=False) # show only legend title
            legend([dummy], [''], loc = 'best', numpoints=3, fontsize = lfs, frameon = 0, fancybox = 0, columnspacing = 1, title='Thermo')
            if save_figs:
                savefig(model_type+'/'+ftype+'/pdfs/HEE/linfit_TD.pdf')


    # Saving FOPT data:
    fname = model_type+'/'+ftype+'/hee/Tc_muT_hee.p'
    pickle.dump(Tc_muT_hee, open(fname, "wb"))
    file.close(open(fname))




####################################   MAIN   #################################

if calculate_hee_grid:
    calculate_HEE_grid()

if plot_hee_grid:
    plot_HEE_grid()

if calculate_hee_lvls:
    calculate_HEE_lvls()

if plot_hee_lvls:
    plot_HEE_lvls()


###########################
print '\ndone: HEE.py\n'
show()