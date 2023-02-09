"""
Calculation of pressure pT4 along narrow T=const lines (as alternative to grid_module_3)
and drawing of the corresponding phase diagram

"""


import numpy as np
import pickle
import math

from fmg_TDprocess import TD_scale, TD_scale_isen, J_calc_fd
from args_and_lambds import args_dic, lambdas_dic
from p_calcer4 import p_calc_line, p_calc_Tlvl, p_PT_calc, p_calc_mulvl
from more_TD import e_and_I

from pylab import * # figure, plot, legend, show, semilogy, scatter, loglog, axis, subplot, xlabel, ylabel, rc, savefig, errorbar, imshow, meshgrid, contour, contourf, colorbar, title, cm, clabel, subplots, grid
#from pylab import figure, plot, legend, show, semilogy, scatter, xlabel, ylabel, rc, axis, savefig, subplot, contour, contourf, colorbar, cm, subplots, grid, title, clabel
from nice_tcks import nice_ticks
import numpy.ma as ma
from scipy.optimize import brentq
from scipy.interpolate import splrep, splev

from matplotlib.path import Path
import matplotlib.patches as patches

#from autograd import grad



#####################
model_type = 'VRY_4'
ftype = args_dic['ftype'][model_type]
V_args = args_dic['V'][model_type]
f_args = args_dic['f'][model_type]
lambdas = lambdas_dic[model_type]
#####################


###### Layout for phase diagrams:
#rc('font', size = 30) #fontsize of axis labels (numbers)
#rc('axes', labelsize = 38, lw = 2) #fontsize of axis labels (symbols)
#rc('xtick.major', pad = 2)
#rc('ytick.major', pad = 7)


#### Layout for small diagrams:
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

lw_for_pd = 10
CEP_dotsize = 35**2
clabel_fs = 18  # fontsize
MS = 5          # marker size
MEW = 1         # marker edge width
lfs = 20        # legend font size
set_labels_manually = 0     ########    <== CHOOSE
if set_labels_manually:
    InlineSpace = 12    # for clabels in PD
else:
    InlineSpace = 5
set_inLine = 1

save_figs = 0
save_FOPT = 0   # save FOPT curve


## plot range:
T_min = 0.5  # *T_CEP
T_max = 1.6 # 1.8
mu_min = 0.5 # 0.0 # *mu_CEP
mu_max = 1.5
scale = 0.1 # masking for mu > mu_max*(1+scale) etc.



### Files
fname = model_type+'/'+ftype+'/TD_gr_'+model_type+'_wmu0.p'
TD_gr = pickle.load(open(fname, "rb"))
file.close(open(fname))

if model_type=='VRY_2':
    fname = model_type+'/'+ftype+'/T_mu_contours_'+model_type+'_Tpts.p'
    Tcont = pickle.load(open(fname, "rb"))
    file.close(open(fname))
if model_type=='no':
    fname = model_type+'/'+ftype+'/T_mu_contours_'+model_type+'.p'
    Tcont = pickle.load(open(fname, "rb"))
    file.close(open(fname))
    nocurves=pickle.load(open('no_curves.p','rb')) # paper read out
if model_type=='VRY_4':
    fname = model_type+'/'+ftype+'/T_mu_contours_'+model_type+'.p'
    Tcont = pickle.load(open(fname, "rb"))
    file.close(open(fname))
    fname = model_type+'/'+ftype+'/T_mu_contours_'+model_type+'_TCEP.p'
    Tcont_TCEP = pickle.load(open(fname, "rb"))
    file.close(open(fname))
    fname = model_type+'/'+ftype+'/nCEP_contour_'+model_type+'.p'
    nCEP_cont = pickle.load(open(fname, "rb"))  # contains TD_slice_scaled along n = n_CEP = const for crit exp calc
    file.close(open(fname))


fname = model_type+'/'+ftype+'/TDTA_'+model_type+'.p'
TDTA = pickle.load(open(fname, "rb"))
file.close(open(fname))


### lattice TD at mu_B = 400 MeV (WuB, O(mu^2), 2012):
fname = 'lattice/TD_lattice_mu400.p'
TD_lattice_mu400 = pickle.load(open(fname, "rb"))
file.close(open(fname))

#+++++++++++++++++++++++ lattice data:
lat = pickle.load(open('QG_latticedata_WuB.p','rb'))
file.close(open('QG_latticedata_WuB.p'))
chi2_lat = pickle.load(open('chi2_wubp.p', "rb"))
file.close(open('chi2_wubp.p'))
chi4_lat = pickle.load(open('chi4_wubp.p', "rb"))
file.close(open('chi4_wubp.p'))
#+++++++++++++++++++++++


### Readout
TD_grid     = TD_gr[0]
TD_grid     = TD_scale(TD_grid, lambdas)[0]
TD_grid_m   = 1*TD_grid   # masked only to set labels in PD
phiPhi_grid = TD_gr[1]
TD_full     = TD_gr[2]
T_levels    = np.sort(Tcont['T'].keys())
mu_levels   = np.sort(Tcont['mu'].keys())
num_Tlvls   = len(T_levels)
num_mupts   = len(Tcont['T'][T_levels[0]][1][:,0])
print 'T_levels, mu_levels: ', T_levels, mu_levels
numerical_anomaly = 93.25
muT_nCEP_tck = splrep(nCEP_cont[:,0], nCEP_cont[:,2]) # = mu(T) on n_CEP curve mimicking FOPT curve



### Thermodynamics on T-axis
TD_Tax = TDTA[0]
phi0_raster = TDTA[1]
Phiphi_T0 = np.vstack((np.zeros(len(phi0_raster)), phi0_raster))
p_Tax = p_calc_line([TD_Tax[0,0], 0], [TD_Tax[-1,0],0], TD_Tax, Phiphi_T0)
print 'p_Tax =', p_Tax


### TD on mu levels: [T, pT4, IT4]
p_on_mulvl400 = [np.zeros(num_Tlvls), np.zeros(num_Tlvls), np.zeros(num_Tlvls)]
p_on_mulvl600 = [np.zeros(num_Tlvls), np.zeros(num_Tlvls), np.zeros(num_Tlvls)]
p_on_mulvl610 = [np.zeros(num_Tlvls), np.zeros(num_Tlvls), np.zeros(num_Tlvls)]
p_on_mulvl800 = [np.zeros(num_Tlvls), np.zeros(num_Tlvls), np.zeros(num_Tlvls)]
p_on_mulvl990 = [np.zeros(num_Tlvls), np.zeros(num_Tlvls), np.zeros(num_Tlvls)]
p_on_mulvl1200 = [np.zeros(num_Tlvls), np.zeros(num_Tlvls), np.zeros(num_Tlvls)]


###++++++++++++++++++++++++         determine phases along mu levels:
for i in range(len(mu_levels)):
    lvl = mu_levels[i]
    acc_cntrTD = Tcont['mu'][lvl]
    phiPhi_slice = acc_cntrTD[0]
    TD_slice_scaled = TD_scale_isen(acc_cntrTD[1], lambdas)[0]

    PT_type = 'ifl'
    for j in range(1, len(TD_slice_scaled[:,0])):
        if TD_slice_scaled[:,0][::-1][j] < TD_slice_scaled[:,0][::-1][j-1]:
            PT_type = '1st'
            break
    print 'mu_const, phase: ', lvl, PT_type



###++++++++++++++++++++++++         pressure and chis along narrow T=const levels for PD
T_grid   = np.ones(( num_Tlvls, num_mupts ))*(-1.0)
s_grid   = np.ones(( num_Tlvls, num_mupts ))*(-1.0)
mu_grid  = np.ones(( num_Tlvls, num_mupts ))*(-1.0)
n_grid   = np.ones(( num_Tlvls, num_mupts ))*(-1.0)
pT4_grid = np.ones(( num_Tlvls, num_mupts ))*(-1.0)
IT4_grid = np.ones(( num_Tlvls, num_mupts ))*(-1.0)

chi2_array = np.zeros(num_Tlvls)
chi2_mu400_array = np.zeros(num_Tlvls)
chi2_mu600_array = np.zeros(num_Tlvls)
chi3_array = np.zeros(num_Tlvls)
chi4_array = np.zeros(num_Tlvls)
chi4_mu400_array = np.zeros(num_Tlvls)
chi4_mu600_array = np.zeros(num_Tlvls)
chi6_array = np.zeros(num_Tlvls)
chi8_array = np.zeros(num_Tlvls)

def fct_spline(x):
    print 'x = ', x
    return splev(np.array([x.value]), splrep(np.array(mu_help), np.array(rho_help)))

def elementwise_grad(fun):
    return grad(lambda x: np.sum(fun(x)))


### loop over T-levels
for i in range(0, num_Tlvls):
    lvl = T_levels[i]
    acc_cntrTD = Tcont['T'][lvl]
    phiPhi_slice = acc_cntrTD[0]

    ## TD:
    TD_slice_scaled = TD_scale_isen(acc_cntrTD[1], lambdas)[0]
    #print 'TD_slice_scaled: ', TD_slice_scaled
    T_grid[i]  = TD_slice_scaled[:,0]
    s_grid[i]  = TD_slice_scaled[:,1]
    mu_grid[i] = TD_slice_scaled[:,2]
    n_grid[i]  = TD_slice_scaled[:,3]

    p_raster = p_calc_Tlvl(p_Tax, TD_slice_scaled, phiPhi_slice, lvl)
    pT4_grid[i] = p_raster[0]/TD_slice_scaled[:,0]**4.0
    IT4_grid[i] = e_and_I(TD_slice_scaled, p_raster[0])[1]/TD_slice_scaled[:,0]**4.0

    ## chis:
    if lvl < 120.0:
        smoothing = 100.0
    elif lvl >= 120.0 and lvl < 160.0:
        smoothing = 0.38
    else:
        smoothing = 2.37
    mu_help, rho_help = zip(*sorted(zip(TD_slice_scaled[:, 2], TD_slice_scaled[:, 3])))
    chi2_array[i] = splev(np.array([0.0]), splrep(mu_help, rho_help), der=1)
    chi2_mu400_array[i] = splev(np.array([400.0]), splrep(mu_help, rho_help), der=1)
    chi2_mu600_array[i] = splev(np.array([600.0]), splrep(mu_help, rho_help), der=1)
    chi3_array[i] = splev(np.array([0.0]), splrep(mu_help, rho_help, s=smoothing), der=2)
    chi4_array[i] = splev(np.array([0.0]), splrep(mu_help, rho_help, s=smoothing), der=3)
    chi4_mu400_array[i] = splev(np.array([400.0]), splrep(mu_help, rho_help, s=smoothing), der=3)
    chi4_mu600_array[i] = splev(np.array([600.0]), splrep(mu_help, rho_help, s=smoothing), der=3)
    chi6_array[i] = splev(np.array([0.0]), splrep(mu_help, rho_help, k=5, s=0.38), der=5)
    #chi8_array[i] = splev(np.array([0.0]), splrep(mu_help, rho_help, s=0.5), der=7)

    #grad_chi2 = elementwise_grad(fct_spline)
    #chi2_array[i] = grad_chi2(0.0)
    #print 'T, chi2T2(mu=0) = ', lvl, chi2_array[i]

    if (model_type=='VRY_2' or model_type=='VRY_4') and lvl==numerical_anomaly:   # harcoded fix of numerical anomaly for T~94 MeV
        indprob = i
        print 'T=94, mu, pT4: ', mu_grid[i], pT4_grid[i]
        print 'T=prob-1, mu, pT4: ', mu_grid[i-1], pT4_grid[i-1]
        figure(0)
        plot(mu_grid[indprob-2], pT4_grid[indprob-2],label='indprob - 2')
        plot(mu_grid[indprob-1], pT4_grid[indprob-1],label='indprob - 1')
        plot(mu_grid[indprob], pT4_grid[indprob],label='indprob')
        T_grid[i] = ma.masked
        mu_grid[i] = ma.masked # mu_grid[i-1]
        pT4_grid[i] = ma.masked # pT4_grid[i-1]
        IT4_grid[i] = ma.masked

    #print 'p_raster = ', p_raster
    #print 'mu_raster = ', p_raster[2], len(p_raster[2])
    print 'T, pT4_0 = ', lvl, p_raster[0][0]/lvl**4.0
    #plot(p_raster[2], p_raster[0]/lvl**4.0)
    #plot(p_raster[2], IT4_grid[i])
    #plot(TD_slice_scaled[:,2], TD_slice_scaled[:,3])
    #show()


if model_type=='VRY_2' or model_type=='VRY_4' and T_levels.min()<numerical_anomaly:
    plot(mu_grid[indprob+1], pT4_grid[indprob+1],label='indprob + 1')
    plot(mu_grid[indprob+2], pT4_grid[indprob+2],label='indprob + 2')
    legend(loc='best')


############## chi_2/T^2:
figure(6)
errorbar(chi2_lat[:,0], chi2_lat[:,1], chi2_lat[:,2], ls='', marker='s', ms=MS, mew=MEW, mec='k', ecolor='grey', color='k', label = 'WuBp lattice')
plot(np.array(T_levels), chi2_array/np.array(T_levels)**2, label=r'$\mu=0$ MeV', c='b')
plot(np.array(T_levels), chi2_mu400_array/np.array(T_levels)**2, label=r'$\mu=400$ MeV', c='g', ls='--')
plot(np.array(T_levels), chi2_mu600_array/np.array(T_levels)**2, label=r'$\mu=600$ MeV', c='r', ls='-.')
axis([100, 600, 0, 0.5])
xlabel(r'$T \, [MeV\,]$')
ylabel(r'$\chi_2 / T^{\, 2}$')
legend(frameon = 0, loc = 'best', numpoints=3)
nice_ticks()
if save_figs:
    savefig(model_type+'/'+ftype+'/pdfs/chis/chi2T2_'+model_type+'_num.pdf')


############## chi_3/T:
figure(7)
plot(np.array(T_levels), chi3_array/np.array(T_levels))
xlabel(r'$T$ [MeV]')
ylabel(r'$\chi_3 / T$ ')


############## chi_4:
figure(8)
errorbar(chi4_lat[:,0], chi4_lat[:,1], chi4_lat[:,2], ls='', marker='s', ms=4, mew=MEW, mec='k', ecolor='grey', color='k', label = 'WuBp lattice')
plot(np.array(T_levels), chi4_array, label=r'$\mu=0$ MeV', c='b')
plot(np.array(T_levels), chi4_mu400_array, label=r'$\mu=400$ MeV', c='g', ls='--')
plot(np.array(T_levels), chi4_mu600_array, label=r'$\mu=600$ MeV', c='r', ls='-.')
axis([100, 600, 0, 0.1])
xlabel(r'$T \, [MeV\,]$')
ylabel(r'$\chi_4$')
nice_ticks()
if save_figs:
    savefig(model_type+'/'+ftype+'/pdfs/chis/chi4_'+model_type+'_num.pdf')


############## chi_6/chi_2:
figure(9)
plot(np.array(T_levels), chi6_array/chi2_array, label=r'$\mu=0$ MeV', c='b')
#axis([100, 600, 0, 0.1])
xlabel(r'$T \, [MeV\,]$')
ylabel(r'$\chi_6 / \chi_2$')
nice_ticks()
if save_figs:
    savefig(model_type+'/'+ftype+'/pdfs/chis/chi6chi2_'+model_type+'_num.pdf')


############## chi_8:
#figure(10)


show()
#############################





###++++++++++++++++++++++++         Phase Structure:
if model_type=='VRY_2':
    T_CEP = 111.5
    mu_CEP = 988.9
if model_type=='VRY_4':
    T_CEP = 111.5
    mu_CEP = 611.5

Tc_list = []        # lists for TD along FOPT curve
muc_list = []
pT4c_list = []
pc_list = []
ln_t = []           # ln(delta T) on FOPT curve for crit exp beta
ln_delta_n = []     # ln(delta n) -"-
ln_t_gamma = []     # ln(delta T) on n_CEP curve for crit exp gamma
ln_chi2 = []        # ln(chi_2) -"-


def tdv_PT_finder(tdv, ltp_tck, htp_tck):
    """Calculation of mu_c for phase transition"""
    return splev(tdv, htp_tck) - splev(tdv, ltp_tck)

## masking:
print '\n'
for i in range(0, num_Tlvls):
    PT_type = 'ifl'
    for j in range(1, num_mupts):
        if mu_grid[i,j] < mu_grid[i,j-1]:
            PT_type = '1st'
            ltp_maxind = j-1
            break
    print 'T, PT_type: ', T_levels[i], PT_type

    if PT_type=='1st':
        for j in range(1, num_mupts)[::-1]:
            if mu_grid[i,j] < mu_grid[i,j-1]:
                htp_minind = j
                break
        print 'T, ltp_maxind, htp_minind: ', T_levels[i], ltp_maxind, htp_minind
        #figure(i+1)
        #plot(mu_grid[i], pT4_grid[i], lw=1, c='b')
        #plot(mu_grid[i,ltp_maxind], pT4_grid[i,ltp_maxind], ls='', marker='s', c='r')
        #plot(mu_grid[i,htp_minind], pT4_grid[i,htp_minind], ls='', marker='s', c='r')
        #show()

        ## mask total instable branch:
        T_grid[i,ltp_maxind+1:htp_minind] = ma.masked
        mu_grid[i,ltp_maxind+1:htp_minind] = ma.masked
        pT4_grid[i,ltp_maxind+1:htp_minind] = ma.masked
        IT4_grid[i,ltp_maxind+1:htp_minind] = ma.masked

        ## spline-determination of mu_c:
        pT4_ltp_tck = splrep(mu_grid[i,0:ltp_maxind], pT4_grid[i,0:ltp_maxind])
        pT4_htp_tck = splrep(mu_grid[i,htp_minind:], pT4_grid[i,htp_minind:])
        #plot(mu_grid[i,0:ltp_maxind], splev(mu_grid[i,0:ltp_maxind], pT4_ltp_tck), ls='-.', c='g', lw=2, label='pT4 ltp')
        #plot(mu_grid[i,htp_minind:], splev(mu_grid[i,htp_minind:], pT4_htp_tck), ls='-.', c='g', lw=2, label='pT4 htp')
        #legend()
        mu_c = brentq(tdv_PT_finder, mu_grid[i,htp_minind], mu_grid[i,ltp_maxind], xtol=1e-12, rtol=1e-10, args = (pT4_ltp_tck, pT4_htp_tck))
        pT4_muc = splev(mu_c, pT4_ltp_tck)  # = p/T^4(mu_c)
        print 'mu_c, pT4_muc = ', mu_c, pT4_muc
        #show()

        ## add TD on FOPT curve:
        if not (T_levels[i]>93.0 and T_levels[i]<94.5):   # excluded range around anomaly
            Tc_list.append(T_levels[i])
            muc_list.append(mu_c)
            pT4c_list.append(pT4_muc)
            pc_list.append(pT4_muc*T_levels[i]**4.0)

        ## condition for removal of ltp point:
        for j in range(0, ltp_maxind+1):
            if mu_grid[i,j]>mu_c:
                ind_muc_ltp = j
                break
        mu_grid[i,ind_muc_ltp] = mu_c           # add (mu_c, pT4_muc) point
        pT4_grid[i,ind_muc_ltp] = pT4_muc       # from spline interpolation
        s_mu_c_ltp = splev(mu_c, splrep(mu_grid[i,0:ltp_maxind], s_grid[i,0:ltp_maxind]))   # s(mu_c) in ltp branch
        n_mu_c_ltp = splev(mu_c, splrep(mu_grid[i,0:ltp_maxind], n_grid[i,0:ltp_maxind]))   # n(mu_c) in ltp branch
        s_grid[i,ind_muc_ltp] = s_mu_c_ltp      # add s(mu_c) ltp
        n_grid[i,ind_muc_ltp] = n_mu_c_ltp      # add n(mu_c) ltp
        T_grid[i,ind_muc_ltp+1:ltp_maxind+1] = ma.masked
        mu_grid[i,ind_muc_ltp+1:ltp_maxind+1] = ma.masked
        pT4_grid[i,ind_muc_ltp+1:ltp_maxind+1] = ma.masked
        IT4_grid[i,ind_muc_ltp+1:ltp_maxind+1] = ma.masked

        ## condition for removal of htp point:
        for j in range(htp_minind, num_mupts)[::-1]:
            if mu_grid[i,j]<mu_c:
                ind_muc_htp = j
                break
        mu_grid[i,ind_muc_htp] = mu_c           # add (mu_c, pT4_muc) point
        pT4_grid[i,ind_muc_htp] = pT4_muc       # from spline interpolation
        s_mu_c_htp = splev(mu_c, splrep(mu_grid[i,htp_minind:], s_grid[i,htp_minind:]))     # s(mu_c) in htp branch
        n_mu_c_htp = splev(mu_c, splrep(mu_grid[i,htp_minind:], n_grid[i,htp_minind:]))     # n(mu_c) in htp branch
        s_grid[i,ind_muc_htp] = s_mu_c_htp      # add s(mu_c) htp
        n_grid[i,ind_muc_htp] = n_mu_c_htp      # add n(mu_c) htp
        T_grid[i,htp_minind:ind_muc_htp] = ma.masked
        mu_grid[i,htp_minind:ind_muc_htp] = ma.masked
        pT4_grid[i,htp_minind:ind_muc_htp] = ma.masked
        IT4_grid[i,htp_minind:ind_muc_htp] = ma.masked
        #plot(mu_grid[i], pT4_grid[i], ls='', marker='s', c='b')
        #show()

        ## data for critical exponent beta:
        if T_levels[i] > T_CEP-15.0:
            ln_t.append(np.log(np.abs((T_levels[i]-T_CEP)/T_CEP)))
            ln_delta_n.append(np.log(n_mu_c_htp-n_mu_c_ltp))

        ## pT4 and IT4 for mu levels:
        #print 'mu, p restricted: ', mu_grid[i][(mu_grid[i]<mu_c)&(mu_grid[i]>0)], pT4_grid[i][(mu_grid[i]<mu_c)&(mu_grid[i]>0)]
        p_on_mulvl400[0][i] = p_on_mulvl600[0][i] = p_on_mulvl610[0][i] = p_on_mulvl800[0][i] = p_on_mulvl990[0][i] = p_on_mulvl1200[0][i] = T_levels[i] # T
        p_on_mulvl400[1][i]  = splev(400.0,  splrep(mu_grid[i][(mu_grid[i]<mu_c)&(mu_grid[i]>0)], pT4_grid[i][(mu_grid[i]<mu_c)&(mu_grid[i]>0)])) # pT4
        p_on_mulvl400[2][i]  = splev(400.0,  splrep(mu_grid[i][(mu_grid[i]<mu_c)&(mu_grid[i]>0)], IT4_grid[i][(mu_grid[i]<mu_c)&(mu_grid[i]>0)])) # IT4
        if model_type=='VRY_2':
            p_on_mulvl600[1][i]  = splev(600.0,  splrep(mu_grid[i][(mu_grid[i]<mu_c)&(mu_grid[i]>0)], pT4_grid[i][(mu_grid[i]<mu_c)&(mu_grid[i]>0)]))
            p_on_mulvl600[2][i]  = splev(600.0,  splrep(mu_grid[i][(mu_grid[i]<mu_c)&(mu_grid[i]>0)], IT4_grid[i][(mu_grid[i]<mu_c)&(mu_grid[i]>0)]))
            p_on_mulvl800[1][i]  = splev(800.0,  splrep(mu_grid[i][(mu_grid[i]<mu_c)&(mu_grid[i]>0)], pT4_grid[i][(mu_grid[i]<mu_c)&(mu_grid[i]>0)]))
            p_on_mulvl800[2][i]  = splev(800.0,  splrep(mu_grid[i][(mu_grid[i]<mu_c)&(mu_grid[i]>0)], IT4_grid[i][(mu_grid[i]<mu_c)&(mu_grid[i]>0)]))
            p_on_mulvl990[1][i]  = splev(990.0,  splrep(mu_grid[i][(mu_grid[i]<mu_c)&(mu_grid[i]>0)], pT4_grid[i][(mu_grid[i]<mu_c)&(mu_grid[i]>0)]))
            p_on_mulvl990[2][i]  = splev(990.0,  splrep(mu_grid[i][(mu_grid[i]<mu_c)&(mu_grid[i]>0)], IT4_grid[i][(mu_grid[i]<mu_c)&(mu_grid[i]>0)]))
            if T_levels[i]>0.88*T_CEP:
                p_on_mulvl1200[1][i] = splev(1200.0, splrep(mu_grid[i][mu_grid[i]>mu_c], pT4_grid[i][mu_grid[i]>mu_c]))
                p_on_mulvl1200[2][i] = splev(1200.0, splrep(mu_grid[i][mu_grid[i]>mu_c], IT4_grid[i][mu_grid[i]>mu_c]))
            else:
                p_on_mulvl1200[1][i] = splev(1200.0, splrep(mu_grid[i][(mu_grid[i]<mu_c)&(mu_grid[i]>0)], pT4_grid[i][(mu_grid[i]<mu_c)&(mu_grid[i]>0)]))
                p_on_mulvl1200[2][i] = splev(1200.0, splrep(mu_grid[i][(mu_grid[i]<mu_c)&(mu_grid[i]>0)], IT4_grid[i][(mu_grid[i]<mu_c)&(mu_grid[i]>0)]))
        if model_type=='VRY_4':
            p_on_mulvl610[1][i]  = splev(610.0,  splrep(mu_grid[i][(mu_grid[i]<mu_c)&(mu_grid[i]>0)], pT4_grid[i][(mu_grid[i]<mu_c)&(mu_grid[i]>0)]))
            p_on_mulvl610[2][i]  = splev(610.0,  splrep(mu_grid[i][(mu_grid[i]<mu_c)&(mu_grid[i]>0)], IT4_grid[i][(mu_grid[i]<mu_c)&(mu_grid[i]>0)]))
            if T_levels[i]>0.88*T_CEP:
                p_on_mulvl800[1][i] = splev(800.0, splrep(mu_grid[i][mu_grid[i]>mu_c], pT4_grid[i][mu_grid[i]>mu_c]))
                p_on_mulvl800[2][i] = splev(800.0, splrep(mu_grid[i][mu_grid[i]>mu_c], IT4_grid[i][mu_grid[i]>mu_c]))
            else:
                p_on_mulvl800[1][i] = splev(800.0, splrep(mu_grid[i][(mu_grid[i]<mu_c)&(mu_grid[i]>0)], pT4_grid[i][(mu_grid[i]<mu_c)&(mu_grid[i]>0)]))
                p_on_mulvl800[2][i] = splev(800.0, splrep(mu_grid[i][(mu_grid[i]<mu_c)&(mu_grid[i]>0)], IT4_grid[i][(mu_grid[i]<mu_c)&(mu_grid[i]>0)]))
    else: # crossover PT_type for T > T_CEP
        p_on_mulvl400[0][i] = p_on_mulvl600[0][i] = p_on_mulvl610[0][i] = p_on_mulvl800[0][i] = p_on_mulvl990[0][i] = p_on_mulvl1200[0][i] = T_levels[i] # T
        p_on_mulvl400[1][i]  = splev(400.0,  splrep(mu_grid[i], pT4_grid[i])) # pT4
        p_on_mulvl400[2][i]  = splev(400.0,  splrep(mu_grid[i], IT4_grid[i])) # IT4
        if model_type=='VRY_2':
            p_on_mulvl600[1][i]  = splev(600.0,  splrep(mu_grid[i], pT4_grid[i]))
            p_on_mulvl600[2][i]  = splev(600.0,  splrep(mu_grid[i], IT4_grid[i]))
            p_on_mulvl800[1][i]  = splev(800.0,  splrep(mu_grid[i], pT4_grid[i]))
            p_on_mulvl800[2][i]  = splev(800.0,  splrep(mu_grid[i], IT4_grid[i]))
            p_on_mulvl990[1][i]  = splev(990.0,  splrep(mu_grid[i], pT4_grid[i]))
            p_on_mulvl990[2][i]  = splev(990.0,  splrep(mu_grid[i], IT4_grid[i]))
            p_on_mulvl1200[1][i] = splev(1200.0, splrep(mu_grid[i], pT4_grid[i]))
            p_on_mulvl1200[2][i] = splev(1200.0, splrep(mu_grid[i], IT4_grid[i]))
        if model_type=='VRY_4':
            p_on_mulvl610[1][i]  = splev(610.0,  splrep(mu_grid[i], pT4_grid[i]))
            p_on_mulvl610[2][i]  = splev(610.0,  splrep(mu_grid[i], IT4_grid[i]))
            p_on_mulvl800[1][i]  = splev(800.0,  splrep(mu_grid[i], pT4_grid[i]))
            p_on_mulvl800[2][i]  = splev(800.0,  splrep(mu_grid[i], IT4_grid[i]))

        ## data for critical exponent gamma:
        if T_levels[i] < T_CEP+20.0:
            mu_proj = splev(T_levels[i], muT_nCEP_tck) # corresponding mu for current T on n_CEP curve mimicking FOPT curve
            chi_2 = float(splev(mu_proj, splrep(mu_grid[i], n_grid[i]), der=1))
            ln_chi2.append(np.log(chi_2))
            ln_t_gamma.append(np.log(np.abs((T_levels[i]-T_CEP)/T_CEP)))

    ## masking outside plot range for good plot:
    for j in range(0, num_mupts):
        if model_type!='no' and (pT4_grid[i,j]<=0 or T_grid[i,j]<(1-scale)*T_min*T_CEP or T_grid[i,j]>(1+scale)*T_max*T_CEP
                                                  or mu_grid[i,j]<(1-scale)*mu_min*mu_CEP or mu_grid[i,j]>(1+scale)*mu_max*mu_CEP):
            T_grid[i,j] = ma.masked
            mu_grid[i,j] = ma.masked
            pT4_grid[i,j] = ma.masked
            IT4_grid[i,j] = ma.masked






### ---------------------------   critical exponents   ------------------------
def fct_scalar(x, y_tck):
    """Calculation of inflection pt via 2nd derivative """
    return splev(x, y_tck, der=2)


###   delta n ~ (T - T_CEP)^beta along FOPT curve
figure(100)
ln_t = np.array(ln_t)
ln_delta_n = np.array(ln_delta_n)

linfit_beta = np.polyfit(ln_t, ln_delta_n, 1)
linfct_beta = np.poly1d(linfit_beta)
print '\ncritical exponent beta, linear fit: ', linfit_beta

plot(ln_t, ln_delta_n, ls='', marker='s')
plot(ln_t, linfct_beta(ln_t))
text(-3, 13, r'$\beta \approx %1.4f$' %linfit_beta[0])
xlabel(r'$\ln(t)$')
ylabel(r'$\ln(\Delta n)$')
nice_ticks()

if save_figs:
    savefig(model_type+'/'+ftype+'/pdfs/critexp/'+model_type+'_critexp_beta.pdf')


###   (n - n_CEP) ~ (mu - mu_CEP)^1/delta along T=T_CEP
num_dist = 18        # minimum index away from CEP which is used for crit exp fit
num_crit_pts = 140   # maximum index away from CEP which is used for crit exp fit
acc_cntrTD = Tcont_TCEP['T'][Tcont_TCEP['T'].keys()[0]]
TD_slice_scaled = TD_scale_isen(acc_cntrTD[1], lambdas)[0]
mu_TCEP = TD_slice_scaled[:,2]
n_TCEP  = TD_slice_scaled[:,3]

mun_tck = splrep(n_TCEP, mu_TCEP) # = mu(n)
dmudn = splev(n_TCEP, mun_tck, der=1)
n_c = brentq(fct_scalar, 4e5, 6e5, xtol=1e-12, rtol=1e-10, args=(mun_tck,)) # n at inflection pt
mu_c = splev(n_c, mun_tck)
ind_CEP = np.amax(np.where(n_TCEP<=n_c))
print 'mu_c, n_c: ', mu_c, n_c

ln_mu_above_fit1 = np.log(np.abs(mu_TCEP - mu_c))[ind_CEP+1:ind_CEP+num_dist]               # for mu > mu_CEP fit, range 1 up to dist away from CEP
ln_n_above_fit1  = np.log(np.abs(n_TCEP - n_c))[ind_CEP+1:ind_CEP+num_dist]
ln_mu_above_fit2 = np.log(np.abs(mu_TCEP - mu_c))[ind_CEP+num_dist:ind_CEP+num_crit_pts]    # for mu > mu_CEP fit, range 2 starting dist away from CEP
ln_n_above_fit2  = np.log(np.abs(n_TCEP - n_c))[ind_CEP+num_dist:ind_CEP+num_crit_pts]
ln_mu_above_plot = np.log(np.abs(mu_TCEP - mu_c))[ind_CEP+1:ind_CEP+num_crit_pts]           # for plot, starting from CEP
ln_n_above_plot  = np.log(np.abs(n_TCEP - n_c))[ind_CEP+1:ind_CEP+num_crit_pts]
linfit_above1 = np.polyfit(ln_mu_above_fit1, ln_n_above_fit1, 1)
linfct_above1 = np.poly1d(linfit_above1)
linfit_above2 = np.polyfit(ln_mu_above_fit2, ln_n_above_fit2, 1)
linfct_above2 = np.poly1d(linfit_above2)
delta1 = 1.0/linfit_above1[0]
delta2 = 1.0/linfit_above2[0]
print 'critical exponent delta, linear fit range 1: ', linfit_above1, delta1
print 'critical exponent delta, linear fit range 2: ', linfit_above2, delta2, '\n'

figure(101)
plot(mu_TCEP, n_TCEP, c='C0') # = n(mu) at T=T_CEP
plot(mu_c, n_c, ls='', marker='s', c='C3')
plot(mu_TCEP[ind_CEP+1:ind_CEP+num_dist], n_TCEP[ind_CEP+1:ind_CEP+num_dist], ls='', marker='s', c='C2')
plot(mu_TCEP[ind_CEP+num_dist:ind_CEP+num_crit_pts], n_TCEP[ind_CEP+num_dist:ind_CEP+num_crit_pts], ls='', marker='s', c='C1')
xlabel(r'$\mu$')
ylabel(r'$n$')
nice_ticks()

figure(102)
axvline(x=ln_mu_above_fit2[0], c='grey', ls=':')
plot(ln_mu_above_plot, ln_n_above_plot, ls='', marker='s', c='C0')
plot(ln_mu_above_fit1, linfct_above1(ln_mu_above_fit1), c='C2')
plot(ln_mu_above_plot, linfct_above2(ln_mu_above_plot), c='C1')
text(-7, 6, r'$\delta \approx %1.4f$' %delta1)
text(-1, 10.5, r'$\delta \approx %1.4f$' %delta2)
xlabel(r'$\ln(\mu-\mu_{CEP})$')
ylabel(r'$\ln(n-n_{CEP})$')
nice_ticks()

if save_figs:
    savefig(model_type+'/'+ftype+'/pdfs/critexp/'+model_type+'_critexp_delta.pdf')


###   chi_2 ~ (T - T_CEP)^-gamma along FOPT curve mimicked by n_CEP curve
figure(103)
ln_t_gamma = ma.asarray(ln_t_gamma)
ln_chi2 = ma.asarray(ln_chi2)
print '\nln_t, ln_chi2: ', ln_t_gamma, ln_chi2

for i in range(len(ln_chi2)):
    if math.isnan(ln_chi2[i]):
        ln_t_gamma[i] = ma.masked
        ln_chi2[i] = ma.masked
ln_t_gamma = ln_t_gamma[~ln_t_gamma.mask]
ln_chi2 = ln_chi2[~ln_chi2.mask]

linfit_gamma = np.polyfit(ln_t_gamma, ln_chi2, 1)
linfct_gamma = np.poly1d(linfit_gamma)
print 'critical exponent gamma, linear fit: ', linfit_gamma, '\n'

plot(ln_t_gamma, ln_chi2, ls='', marker='s')
plot(ln_t_gamma, linfct_gamma(ln_t_gamma))
gamma = -linfit_gamma[0]
text(-3, 10.2, r'$\gamma \approx %1.4f$' %gamma)
xlabel(r'$\ln(t)$')
ylabel(r'$\ln(\chi_2)$')
nice_ticks()

if save_figs:
    savefig(model_type+'/'+ftype+'/pdfs/critexp/'+model_type+'_critexp_gamma.pdf')





### ---------------------------   Plotting PHASE DIAGRAMS   -------------------
## plot parameters:
if model_type=='VRY_2':
    ## pT4:
    tick_max = 12.0
    div_lim = 2.0
    ff = 1.7
    pT4_lvls = np.hstack(( np.linspace(0.0, div_lim, 200), np.linspace(div_lim*1.001, tick_max, 200) ))
    pT4_ticks = np.arange(0, tick_max*1.001, 2.0)

    pT4_levels_2a = np.arange(0.0, 2.51, 0.5)
    pT4_levels_2b = np.arange(3.0, 11.0, 1.0)
    pT4_levels_2c = np.array([0.7])
    pT4_levels_2 = np.sort(np.hstack(( pT4_levels_2a, pT4_levels_2b, pT4_levels_2c )))

    pT4_colors_2 = [None]*len(pT4_levels_2)
    for i in range(0, len(pT4_colors_2)):
        pT4_colors_2[i] = cm.Greens(pT4_levels_2[i]/np.amax(pT4_levels_2))

    ## sn:
    tick_max = 26.0
    div_lim = 15.0
    sn_lvls = np.hstack(( np.linspace(1.0, div_lim, 300), np.linspace(div_lim*1.001, tick_max, 300) ))
    sn_ticks = list(np.arange(0, tick_max*1.001, tick_max/10.0))

    sn_levels_2 = np.sort(np.array([4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0, 20.0]))
    sn_labelpos = [(1.44, 0.92), (1.37, 1.02), (1.25, 1.14), (1.14, 1.21), (1.04, 1.26), (0.97, 1.32), (1.42, 0.58), (0.89, 1.35),
                   (0.77, 1.38), (1.23, 0.65), (1.08, 0.68), (0.68, 1.40), (0.94, 0.74), (0.63, 1.27), (0.75, 1.08), (0.73, 0.75), (0.60, 1.14)]

    sn_colors_2 = [None]*len(sn_levels_2)
    for i in range(0, len(sn_colors_2)):
        sn_colors_2[i] = cm.Greens(sn_levels_2[i]/np.amax(sn_levels_2))

    ## sn in T-logn:
    sn_lvls_log = np.hstack(( np.linspace(1.0, div_lim, 300), np.linspace(div_lim*1.001, tick_max, 300) ))
    sn_ticks_log = list(np.arange(5, tick_max*1.001, 5))

    sn_levels_2_log = np.sort(np.array([6.0, 7.0, 8.0, 10.0, 11.5, 13.0, 15.0, 17.0, 20.0, 25.0, 30.0]))
    sn_labelpos_log = [(0.98, 0.69), (1.06, 0.91), (1.05, 1.06), (0.77, 1.18), (0.60, 1.26), (-2.58, 0.54), (0.75, 1.37), (-2.65, 0.58),
                       (-2.71, 0.66), (-0.04, 1.23), (-2.68, 0.73), (-0.52, 1.14), (-0.91, 1.12), (-1.42, 1.11), (-2.17, 1.05)]

    sn_colors_2_log = [None]*len(sn_levels_2_log)
    for i in range(0, len(sn_colors_2_log)):
        sn_colors_2_log[i] = cm.Greens(sn_levels_2_log[i]/np.amax(sn_levels_2_log))

    ## nT3:
    tick_max = 6.0
    div_lim = 1.0
    nT3_lvls = np.hstack(( np.linspace(0.0, div_lim, 300), np.linspace(div_lim*1.001, tick_max, 300) ))
    nT3_ticks = list(np.arange(0, tick_max*1.001, tick_max/10.0))

    nT3_levels_2_a = np.array([0.1, 0.15, 0.2, 0.3])
    nT3_levels_2_b = np.arange(0.4, 1.0, 0.2)
    nT3_levels_2_c = np.arange(1.0, 4.61, 0.4)
    nT3_levels_2 = np.sort(np.hstack((nT3_levels_2_a, nT3_levels_2_b, nT3_levels_2_c)))

    nT3_colors_2 = [None]*len(nT3_levels_2)
    for i in range(0, len(nT3_colors_2)):
        nT3_colors_2[i] = cm.Greens(nT3_levels_2[i]/np.amax(nT3_levels_2))

if model_type=='VRY_4':
    ## pT4:
    tick_max = 5.7
    div_lim = 1.0
    ff = 1.7
    pT4_lvls = np.hstack(( np.linspace(0.0, div_lim, 400), np.linspace(div_lim*1.001, tick_max, 200) ))
    pT4_ticks = list(np.arange(0, tick_max*1.001, 0.8))

    pT4_levels_2a = np.arange(0.0, 5.01, 0.5)
    pT4_levels_2c = np.array([0.45, 0.6, 0.8])
    pT4_levels_2 = np.sort(np.hstack(( pT4_levels_2a, pT4_levels_2c )))

    pT4_colors_2 = [None]*len(pT4_levels_2)
    for i in range(0, len(pT4_colors_2)):
        pT4_colors_2[i] = cm.Greens(pT4_levels_2[i]/np.amax(pT4_levels_2))

    ## sn:
    tick_max = 42.0
    div_lim = 15.0
    sn_lvls = np.hstack(( np.linspace(6.0, div_lim, 300), np.linspace(div_lim*1.001, tick_max, 300) ))
    sn_ticks = list(np.arange(6, tick_max*1.001, 4))

    sn_levels_2 = np.sort(np.array([7.5, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 15.0, 17.0, 20.0, 25.0, 30.0]))
    sn_labelpos = [(1.44, 0.92), (1.37, 1.02), (1.25, 1.14), (1.14, 1.21), (1.04, 1.26), (0.97, 1.32), (1.42, 0.58), (0.89, 1.35),
                   (0.77, 1.38), (1.23, 0.65), (1.08, 0.68), (0.68, 1.40), (0.94, 0.74), (0.63, 1.27), (0.75, 1.08), (0.73, 0.75), (0.60, 1.14)]

    sn_colors_2 = [None]*len(sn_levels_2)
    for i in range(0, len(sn_colors_2)):
        sn_colors_2[i] = cm.Greens(sn_levels_2[i]/np.amax(sn_levels_2))

    ## sn in T-logn:
    sn_lvls_log = np.hstack(( np.linspace(5.0, div_lim, 300), np.linspace(div_lim*1.001, tick_max, 300) ))
    sn_ticks_log = list(np.arange(5, tick_max*1.001, 5))

    sn_levels_2_log = np.sort(np.array([6.0, 7.0, 8.0, 10.0, 11.5, 13.0, 15.0, 17.0, 20.0, 25.0, 30.0]))
    sn_labelpos_log = [(0.98, 0.69), (1.06, 0.91), (1.05, 1.06), (0.77, 1.18), (0.60, 1.26), (-2.58, 0.54), (0.75, 1.37), (-2.65, 0.58),
                       (-2.71, 0.66), (-0.04, 1.23), (-2.68, 0.73), (-0.52, 1.14), (-0.91, 1.12), (-1.42, 1.11), (-2.17, 1.05)]

    sn_colors_2_log = [None]*len(sn_levels_2_log)
    for i in range(0, len(sn_colors_2_log)):
        sn_colors_2_log[i] = cm.Greens(sn_levels_2_log[i]/np.amax(sn_levels_2_log))

    ## nT3:
    tick_max = 4.5
    div_lim = 0.4
    nT3_lvls = np.hstack(( np.linspace(0.0, div_lim, 500), np.linspace(div_lim*1.001, tick_max, 300) ))
    nT3_ticks = list(np.arange(0, tick_max*1.001, 0.5))

    nT3_levels_2_a = np.array([0.07, 0.1, 0.15, 0.2, 0.3])
    nT3_levels_2_b = np.arange(0.4, 1.0, 0.2)
    nT3_levels_2_c = np.arange(1.0, 4.01, 0.4)
    nT3_levels_2 = np.sort(np.hstack((nT3_levels_2_a, nT3_levels_2_b, nT3_levels_2_c)))

    nT3_colors_2 = [None]*len(nT3_levels_2)
    for i in range(0, len(nT3_colors_2)):
        nT3_colors_2[i] = cm.Greens(nT3_levels_2[i]/np.amax(nT3_levels_2))


### Plotting:
if model_type=='VRY_2' or model_type=='VRY_4':
    plot_range = [mu_min, mu_max, T_min, T_max]

    ### --------------------   Plotting scaled pressure p/T^4  --------------------------
    figure(200, figsize = (ff*8.0, ff*6.2))
    contour(mu_grid/mu_CEP, T_grid/T_CEP, pT4_grid, levels=pT4_lvls, linewidths = 10, cmap=cm.jet, zorder=-3)

    nice_ticks()
    axis(plot_range)
    colorbar(spacing = 'proportional', ticks = pT4_ticks)

    cont = contour(mu_grid/mu_CEP, T_grid/T_CEP, pT4_grid, levels=pT4_levels_2, colors=pT4_colors_2, linewidths = 3, zorder=-2)
    clabel(cont, pT4_levels_2, inline=set_inLine, fontsize = clabel_fs, fmt = '%1.2f', manual=set_labels_manually, inline_spacing=InlineSpace)

    plot(np.array(muc_list)/mu_CEP, np.array(Tc_list)/T_CEP, color = 'grey', lw = lw_for_pd, zorder=-1)
    scatter(1.0, 1.0, c='white', s=CEP_dotsize, alpha=1, zorder=1, edgecolor='white')

    xlabel(r'$\mu/\mu_{CEP}$')
    ylabel(r'$T/T_{CEP}$')
    title(r'$p/T^{\, 4}$')
    if save_figs:
        savefig(model_type+'/'+ftype+'/pdfs/PD/'+model_type+'_pressure_PD.pdf')


    ### --------------------   Plotting scaled entropy s/T^3  --------------------------
    ## sT3:
    tick_max = 32.0
    div_lim = 5.0
    if mu_min == 0.0:
        sT3_lvls = np.hstack(( np.linspace(1.0, div_lim, 400), np.linspace(div_lim*1.001, tick_max, 400) ))
    else:
        sT3_lvls = np.hstack(( np.linspace(1.0, div_lim, 300), np.linspace(div_lim*1.001, tick_max, 200) ))
    sT3_ticks = list(np.arange(0, tick_max*1.001, 4))

    if mu_min == 0.0:
        sT3_levels_2_a = np.array([1.5, 1.7, 2.0, 2.5, 3.0, 3.5])
        sT3_levels_2_b = np.arange(4.0, 19.0, 2.0)
        sT3_levels_2_c = np.array([22.0])
    else:
        sT3_levels_2_a = np.array([1.8, 2.0, 2.5, 3.0, 3.5, 4.5])
        sT3_levels_2_b = np.arange(4.0, 20.5, 2.0)
        sT3_levels_2_c = np.array([22.0, 25.0])
    sT3_levels_2 = np.sort(np.hstack((sT3_levels_2_a, sT3_levels_2_b, sT3_levels_2_c)))

    sT3_colors_2 = [None]*len(sT3_levels_2)
    for i in range(0, len(sT3_colors_2)):
        sT3_colors_2[i] = cm.Greens(sT3_levels_2[i]/np.amax(sT3_levels_2))

    ## plotting PD:
    figure(201, figsize = (ff*8.0, ff*6.2))
    contour(mu_grid/mu_CEP, T_grid/T_CEP, s_grid/T_grid**3, levels=sT3_lvls, linewidths = 10, cmap=cm.jet, zorder=-3)

    if mu_min == 0.0:
        ax = subplot(111)
        ax.set_xticks(np.arange(mu_min, mu_max, 0.2))

    nice_ticks()
    axis(plot_range)
    colorbar(spacing = 'proportional', ticks = sT3_ticks)

    cont = contour(mu_grid/mu_CEP, T_grid/T_CEP, s_grid/T_grid**3, levels=sT3_levels_2, colors=sT3_colors_2, linewidths = 3, zorder=-2)
    clabel(cont, sT3_levels_2, inline=set_inLine, fontsize = clabel_fs, fmt = '%1.1f', manual=set_labels_manually, inline_spacing=InlineSpace)

    plot(np.array(muc_list)/mu_CEP, np.array(Tc_list)/T_CEP, color = 'grey', lw = lw_for_pd, zorder=-1)
    scatter(1.0, 1.0, c='white', s=CEP_dotsize, alpha=1, zorder=1, edgecolor='white')

    xlabel(r'$\mu/\mu_{CEP}$')
    ylabel(r'$T/T_{CEP}$')
    title(r'$s/T^{\,3}$')
    if save_figs and mu_min!=0.0:
        savefig(model_type+'/'+ftype+'/pdfs/PD/'+model_type+'_entropy_PD.pdf')
    if save_figs and mu_min==0.0:
        savefig(model_type+'/'+ftype+'/pdfs/PD/'+model_type+'_entropy_PD_excerpt.pdf')


    ### --------------------   Plotting isentropes s/n  ---------------------------
    ## plotting PD:
    figure(202, figsize = (ff*8.0, ff*6.2))
    contour(mu_grid/mu_CEP, T_grid/T_CEP, s_grid/n_grid, levels=sn_lvls, linewidths = 10, cmap=cm.jet, zorder=-3)

    nice_ticks()
    axis(plot_range)
    colorbar(spacing = 'proportional', ticks = sn_ticks)

    cont = contour(mu_grid/mu_CEP, T_grid/T_CEP, s_grid/n_grid, levels=sn_levels_2, colors=sn_colors_2, linewidths = 3, zorder=-2)
    #clabel(cont, sn_levels_2, inline=set_inLine, fontsize = clabel_fs, fmt = '%1.1f', manual = set_labels_manually)

    isens = contour(TD_grid_m[:,:,2]/mu_CEP+0.027, TD_grid_m[:,:,0]/T_CEP, TD_grid_m[:,:,1]/TD_grid_m[:,:,3], levels=sn_levels_2, colors=sn_colors_2, linewidths=3, zorder=-4) # shifted to have no overlapping labels (not visible)
    tl = clabel(isens, sn_levels_2, inline=set_inLine, fontsize = clabel_fs, fmt = '%1.1f', manual = sn_labelpos)
    #for t in tl:
        #t.set_bbox(dict(fc='w', ec='none', alpha=0.5, boxstyle='round'))

    plot(np.array(muc_list)/mu_CEP, np.array(Tc_list)/T_CEP, color = 'grey', lw = lw_for_pd, zorder=-1)
    scatter(1.0, 1.0, c='white', s=CEP_dotsize, alpha=1, zorder=1, edgecolor='white')

    xlabel(r'$\mu/\mu_{CEP}$')
    ylabel(r'$T/T_{CEP}$')
    title(r'$s/n$')
    if save_figs:
        savefig(model_type+'/'+ftype+'/pdfs/PD/'+model_type+'_isentropes_PD.pdf')


    ### --------------------   Plotting scaled density n/T^3  ---------------------
    ## plotting PD:
    figure(203, figsize = (ff*8.0, ff*6.2))
    contour(mu_grid/mu_CEP, T_grid/T_CEP, n_grid/T_grid**3.0, levels=nT3_lvls, linewidths = 10, cmap=cm.jet, zorder=-3)

    nice_ticks()
    axis(plot_range)
    colorbar(spacing = 'proportional', ticks = nT3_ticks)

    cont = contour(mu_grid/mu_CEP, T_grid/T_CEP, n_grid/T_grid**3.0, levels=nT3_levels_2, colors=nT3_colors_2, linewidths = 3, zorder=-2)
    clabel(cont, nT3_levels_2, inline=set_inLine, fontsize = clabel_fs, fmt = '%1.2f', manual=set_labels_manually, inline_spacing=InlineSpace)

    plot(np.array(muc_list)/mu_CEP, np.array(Tc_list)/T_CEP, color = 'grey', lw = lw_for_pd, zorder=-1)
    scatter(1.0, 1.0, c='white', s=CEP_dotsize, alpha=1, zorder=1, edgecolor='white')

    xlabel(r'$\mu/\mu_{CEP}$')
    ylabel(r'$T/T_{CEP}$')
    title(r'$n/T^{\,3}$')
    if save_figs:
        savefig(model_type+'/'+ftype+'/pdfs/PD/'+model_type+'_density_PD.pdf')


    ### --------------------   Plotting isentropes s/n in T-log(n)  ---------------------
    ## plotting PD:
    fig = figure(204, figsize = (ff*8.387, ff*6.55))
    contour(np.log(n_grid/T_CEP**3), T_grid/T_CEP, s_grid/n_grid, levels=sn_lvls_log, linewidths = 10, cmap=cm.jet, zorder=-3)

    nice_ticks()
    axis([-5, 2, T_min, T_max])
    colorbar(spacing = 'proportional', ticks = sn_ticks) # sn_ticks_log

    cont = contour(np.log(n_grid/T_CEP**3), T_grid/T_CEP, s_grid/n_grid, levels=sn_levels_2_log, colors=sn_colors_2_log, linewidths = 3, zorder=-2)
    #clabel(cont, sn_levels_2_log, inline=set_inLine, fontsize = clabel_fs, fmt = '%1.1f', manual=set_labels_manually)

    isens = contour(np.log(TD_grid_m[:,:,3]/T_CEP**3)-0.1, TD_grid_m[:,:,0]/T_CEP, TD_grid_m[:,:,1]/TD_grid_m[:,:,3], levels=sn_levels_2_log, colors=sn_colors_2_log, linewidths=3, zorder=-5) # shifted to have no overlapping labels (not visible)
    tl = clabel(isens, sn_levels_2_log, inline=set_inLine, fontsize = clabel_fs, fmt = '%1.1f', manual = sn_labelpos_log)

    verts = [
        (-2.5, T_min),
        (-2.5, 1.05),
        (0.5, 1.05),
        (0.5, T_min),
        (-2.5, T_min),
        ]
    codes = [Path.MOVETO,
             Path.LINETO,
             Path.LINETO,
             Path.LINETO,
             Path.CLOSEPOLY,
             ]
    path = Path(verts, codes)
    ax = fig.add_subplot(111)
    patch = patches.PathPatch(path, facecolor='grey', lw=2, color='grey', zorder=-4)
    ax.add_patch(patch)

    xlabel(r'$\log \, n/T_{CEP}^{\,3}$')
    ylabel(r'$T/T_{CEP}$')
    title(r'$s/n$')
    if save_figs:
        savefig(model_type+'/'+ftype+'/pdfs/PD/'+model_type+'_coexistence_PD.pdf')


    ### --------------------   Plotting critical pressure on FOPT curve   ---------
    figure(300)
    plot(np.array(Tc_list)/T_CEP, np.array(pc_list))

    xlabel(r'$T/T_{CEP}$')
    ylabel(r'$p(T, \mu_c(T)) \, [MeV^{\,4}]$')
    ax = subplot(111)
    if model_type=='VRY_2':
        axis([0.6, 1.0, 1.0*1e8, 2.2*1e8])
        ax.set_xticks(np.array([0.6, 0.7, 0.8, 0.9, 1.0]))
    elif model_type=='VRY_4':
        axis([T_min, 1.0, 0.35*1e8, 1.2*1e8])
        ax.set_yticks(np.arange(0.4, 1.201, 0.1)*1e8)
    nice_ticks()
    if save_figs:
        savefig(model_type+'/'+ftype+'/pdfs/PD/'+model_type+'_pressure_crit.pdf')




### ------------------------   Plotting pT4 on mu levels   --------------------
figure(400)
errorbar(lat['T'], lat['pT4'], lat['dpT4'], ls='', marker='s', ms=MS, mew=MEW, mec='k', ecolor='grey', color='k', label = 'WuBp lattice')
plot(p_Tax[1], p_Tax[0]/p_Tax[1]**4.0, label = r'$\mu = %d \, MeV$' %0, c='k')
errorbar(TD_lattice_mu400['T'], TD_lattice_mu400['pT4'], TD_lattice_mu400['dpT4'], ls='', marker='s', ms=MS, mew=MEW, mec='r', ecolor='r', color='r')
plot(p_on_mulvl400[0], p_on_mulvl400[1], label = r'$\mu = %d \, MeV$' %400, c='C3')

if model_type=='VRY_2':
    plot(p_on_mulvl600[0], p_on_mulvl600[1], label = r'$\mu = %d \, MeV$' %600, c='b')
    plot(p_on_mulvl800[0], p_on_mulvl800[1], label = r'$\mu = %d \, MeV$' %800, c='orange')
    plot(p_on_mulvl990[0], p_on_mulvl990[1], label = r'$\mu = %d \, MeV$' %990, c='g')
    plot(p_on_mulvl1200[0], p_on_mulvl1200[1], label = r'$\mu = %d \, MeV$' %1200, c='m')

if model_type=='VRY_4':
    plot(p_on_mulvl610[0], p_on_mulvl610[1], label = r'$\mu = %d \, MeV$' %610, c='b')
    plot(p_on_mulvl800[0], p_on_mulvl800[1], label = r'$\mu = %d \, MeV$' %800, c='orange')

if model_type=='no':
    plot(nocurves[400]['pT4'][:,0], nocurves[400]['pT4'][:,1], c='g', label='no paper')

xlabel(r'$T \, [MeV\,]$')
ylabel(r'$p/T^{\,4}$')
axis([100, 500, 0, 7.5])
if model_type=='no' or 'VRY_4':
    axis([100, 500, 0, 4.5])
ax = subplot(111)
ax.set_xticks(np.array([100, 200, 300, 400, 500]))
nice_ticks()
legend(loc = 'best', numpoints=3, fontsize = lfs, frameon = 0, fancybox = 0, columnspacing = 1)
if save_figs:
    savefig(model_type+'/'+ftype+'/pdfs/curves/'+model_type+'_pT4_mulvls.pdf')



### ------------------------   Plotting IT4 on mu levels   --------------------
figure(500)
errorbar(lat['T'],lat['IT4'],lat['dIT4'], ls='', marker='s', ms=MS, mew=MEW, mec='k', ecolor='grey', color='k')
plot(p_Tax[1], e_and_I(TD_Tax, p_Tax[0])[1]/p_Tax[1]**4.0, label = r'$\mu = %d \, MeV$' %0, c='k')
errorbar(TD_lattice_mu400['T'], TD_lattice_mu400['IT4'], TD_lattice_mu400['dIT4'], ls='', marker='s', ms=MS, mew=MEW, mec='r', ecolor='r', color='r')
plot(p_on_mulvl400[0], p_on_mulvl400[2], label = r'$\mu = %d \, MeV$' %400, c='C3')

if model_type=='VRY_2':
    plot(p_on_mulvl600[0], p_on_mulvl600[2], label = r'$\mu = %d \, MeV$' %600, c='b')
    plot(p_on_mulvl800[0], p_on_mulvl800[2], label = r'$\mu = %d \, MeV$' %800, c='orange')
    plot(p_on_mulvl990[0], p_on_mulvl990[2], label = r'$\mu = %d \, MeV$' %990, c='g')
    plot(p_on_mulvl1200[0], p_on_mulvl1200[2], label = r'$\mu = %d \, MeV$' %1200, c='m')

if model_type=='VRY_4':
    plot(p_on_mulvl610[0], p_on_mulvl610[2], label = r'$\mu = %d \, MeV$' %610, c='b')
    plot(p_on_mulvl800[0], p_on_mulvl800[2], label = r'$\mu = %d \, MeV$' %800, c='orange')

if model_type=='no':
    plot(nocurves[400]['IT4'][:,0], nocurves[400]['IT4'][:,1], c='g', label='no paper')

xlabel(r'$T \, [MeV\,]$')
ylabel(r'$I/T^{\,4}$')
axis([100, 500, 0, 18])
if model_type=='no' or 'VRY_4':
    axis([100, 500, 0, 10])
ax = subplot(111)
ax.set_xticks(np.array([100, 200, 300, 400, 500]))
nice_ticks()
#legend(loc = 'best', numpoints=3, fontsize = lfs, frameon = 0, fancybox = 0, columnspacing = 1)
if save_figs:
    savefig(model_type+'/'+ftype+'/pdfs/curves/'+model_type+'_IT4_mulvls.pdf')

#print 'p_on_mulvl400: ', p_on_mulvl400





### ------------------------   s/n  EXCERPT (Comparison to lattice)   ------###
fname = 'lattice/SN_lattice.p'
SN_lattice = pickle.load(open(fname, "rb"))
file.close(open(fname))

levels = ['30', '51', '70', '94', '144', '420']
isen_levels_excerpt = np.array([30.0, 51.0, 70.0, 94.0, 144.0, 420.0])
isen_colors_excerpt2 = ['b', 'g', 'r', 'c', 'darkviolet', 'olive']

figure(600)
isens = contour(TD_grid[:, :, 2], TD_grid[:, :, 0], TD_grid[:, :, 1]/TD_grid[:, :, 3],
                levels = isen_levels_excerpt, colors = isen_colors_excerpt2, linewidths = 4, zorder=-2)
for i in range(0, len(levels)):
    lvl = levels[i]
    plot(SN_lattice['mu_SN'+lvl], SN_lattice['T_SN'+lvl], color=isen_colors_excerpt2[i], ls='', marker='s', ms=MS, mew=MEW, zorder=-1)
nice_ticks()
axis([0, 400, 50, 300])
clabel(isens, isen_levels_excerpt, inline=set_inLine, fontsize = clabel_fs, fmt = '%1.1f', manual=set_labels_manually, zorder=1)

ylabel(r'$T \, [MeV\,]$')
xlabel(r'$\mu \, [MeV\,]$')
title(r'$s/n$')
if save_figs:
    savefig(model_type+'/'+ftype+'/pdfs/PD/'+model_type+'_pd_sn_excerpt_lattice.pdf')




### ------------------------   Saving FOPT data:
if save_FOPT:
    Tc_muT = np.vstack((muc_list, Tc_list))  # <== FOPT information
    fname = model_type+'/'+ftype+'/Tc_muT.p'
    pickle.dump(Tc_muT, open(fname, "wb"))
    file.close(open(fname))















########
show()
###++++++++++++++++++++++++
print '\ndone: pressure_PD.py'