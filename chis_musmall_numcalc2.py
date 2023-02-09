import numpy as np
import math
import pickle
from scipy.misc import derivative as deriv
from scipy.interpolate import splrep, splev
from pylab import figure, plot, show, xlabel, ylabel, legend

from TD_calc_pow import TD_calc_pointwise
from args_and_lambds import args_dic, lambdas_dic
from fmg_TDprocess import TD_scale_isen

model_type = 'VRY_4'
ftype = args_dic['ftype'][model_type]

fname = model_type+'/'+ftype+'/TDTA_'+model_type+'.p'
TDTA = pickle.load(open(fname, "rb"))
file.close(open(fname))

V_args = args_dic['V'][model_type]
f_args = args_dic['f'][model_type]
lambdas = lambdas_dic[model_type]

TD = TDTA[0]
phi0_raster = TDTA[1]
T_raster = TD[:, 0]

print phi0_raster, T_raster

def rho_Phi1r(Phi1r, phi0, V_args, f_args, r_mid):
    global TD_pts, tdp0
    if np.round(phi0, 12) in TD_pts.keys() and np.round(Phi1r, 12) in TD_pts[np.round(phi0, 12)].keys():
        TD = TD_pts[np.round(phi0, 12)][np.round(Phi1r,12)]
    else:
        TD = TD_calc_pointwise(phi0, Phi1r, V_args, f_args, r_mid)#[0]
        if TD==0: # in case of error
            return 0
        TD = TD[0]
        tdp0.update({np.round(Phi1r, 12):TD})
        TD_pts.update({np.round(phi0, 12):tdp0})
    print 'phi0 =', phi0, 'Phi1r =', Phi1r, 'TD =', TD, TD[3], TD[3]/TD[2]
    return TD[3]

def mu_Phi1r(Phi1r, phi0, V_args, f_args, r_mid):
    global TD_pts, tdp0
    if np.round(phi0, 12) in TD_pts.keys() and np.round(Phi1r, 12) in TD_pts[np.round(phi0,12)].keys():
        TD = TD_pts[np.round(phi0, 12)][np.round(Phi1r,12)]
    else:
        TD = TD_calc_pointwise(phi0, Phi1r, V_args, f_args, r_mid)#[0]
        if TD==0: # in case of error
            return 0
        TD = TD[0]
        tdp0.update({np.round(Phi1r, 12):TD})
        TD_pts.update({np.round(phi0, 12):tdp0})
    return TD[2]

def chi_i_calc_musmall(Phi1r, phi0, V_args, f_args, r_mid, order, d_x, npts):
    ## dX denotes dX/d(Phi_1/Phi_1max)
    if (rho_Phi1r(Phi1r, phi0, V_args, f_args, r_mid)==0 or mu_Phi1r(Phi1r, phi0, V_args, f_args, r_mid)==0): # in case of error
        return np.float('nan')
    else:
        drho = deriv(rho_Phi1r, Phi1r, dx = d_x, n = 1, args = (phi0, V_args, f_args, r_mid), order = npts)
        dmu = deriv(mu_Phi1r, Phi1r, dx = d_x, n = 1, args = (phi0, V_args, f_args, r_mid), order = npts)
        if order == 2:
            return drho/dmu
        elif order == 3:
            ddrho = deriv(rho_Phi1r, Phi1r, dx = d_x, n = 2, args = (phi0, V_args, f_args, r_mid), order = npts)
            ddmu = deriv(mu_Phi1r, Phi1r, dx = d_x, n = 2, args = (phi0, V_args, f_args, r_mid), order = npts)
            return ddrho/ddmu# ddrho/dmu**2.0 - drho*ddmu/dmu**3.0
        elif order == 4:
            ddrho = deriv(rho_Phi1r, Phi1r, dx = d_x, n = 2, args = (phi0, V_args, f_args, r_mid), order = npts)
            ddmu = deriv(mu_Phi1r, Phi1r, dx = d_x, n = 2, args = (phi0, V_args, f_args, r_mid), order = npts)
            dddrho = deriv(rho_Phi1r, Phi1r, dx = d_x, n = 3, args = (phi0, V_args, f_args, r_mid), order = npts)
            dddmu = deriv(mu_Phi1r, Phi1r, dx = d_x, n = 3, args = (phi0, V_args, f_args, r_mid), order = npts)
            return dddrho/dddmu# dddrho/dmu**3.0 - (3.0*ddrho*ddmu + drho*dddmu)/dmu**4.0 - 3.0*drho*ddmu**2.0/dmu**5.0

def chi_2_rhomu_ratio(Phi1r, phi0, V_args, f_args, r_mid):
    rho = rho_Phi1r(Phi1r, phi0, V_args, f_args, r_mid)
    mu = mu_Phi1r(Phi1r, phi0, V_args, f_args, r_mid)
    return rho/mu

def chi_3_from_chi2(Phi1r, phi0, V_args, f_args, r_mid, d_x, npts):
    #dchi_2 = deriv(chi_i_calc_musmall, Phi1r, dx = d_x, n = 1, args = (phi0, V_args, f_args, r_mid, 2, d_x, npts))
    dchi_2 = deriv(chi_2_rhomu_ratio, Phi1r, dx = d_x, n = 1, args = (phi0, V_args, f_args, r_mid))
    dmu = deriv(mu_Phi1r, Phi1r, dx = d_x, n = 1, args = (phi0, V_args, f_args, r_mid), order = npts)
    return dchi_2/dmu

def chi_4_from_chi3(Phi1r, phi0, V_args, f_args, r_mid, d_x, npts):
    dchi_3 = deriv(chi_3_from_chi2, Phi1r, dx = d_x, n = 1, args = (phi0, V_args, f_args, r_mid, d_x, npts))
    dmu = deriv(mu_Phi1r, Phi1r, dx = d_x, n = 1, args = (phi0, V_args, f_args, r_mid), order = npts)
    return dchi_3/dmu

def chi_5_from_chi4(Phi1r, phi0, V_args, f_args, r_mid, d_x, npts):
    dchi_4 = deriv(chi_4_from_chi3, Phi1r, dx = d_x, n = 1, args = (phi0, V_args, f_args, r_mid, d_x, npts))
    dmu = deriv(mu_Phi1r, Phi1r, dx = d_x, n = 1, args = (phi0, V_args, f_args, r_mid), order = npts)
    return dchi_4/dmu

def chi_6_from_chi5(Phi1r, phi0, V_args, f_args, r_mid, d_x, npts):
    dchi_5 = deriv(chi_5_from_chi4, Phi1r, dx = d_x, n = 1, args = (phi0, V_args, f_args, r_mid, d_x, npts))
    dmu = deriv(mu_Phi1r, Phi1r, dx = d_x, n = 1, args = (phi0, V_args, f_args, r_mid), order = npts)
    return dchi_5/dmu

TD_pts = {}
tdp0 = {}
#chi_2_raster = np.zeros(len(phi0_raster))
#chi_3_raster = np.zeros(len(phi0_raster))
#chi_4_raster = np.zeros(len(phi0_raster))
#chi_5_raster = np.zeros(len(phi0_raster))
#chi_6_raster = np.zeros(len(phi0_raster))
#chi_4_raster_2 = np.zeros(len(phi0_raster))
d_x = 1e-3
npts = 5
Phi1r_raster = np.array([1e-10, 1e-4, 1e-3])
Phi1r_raster = np.array([1e-6])
chis_dic = {}


for Phi1r in Phi1r_raster:
    print '\n\n\n\n+++Phi1r = ', Phi1r
    r_mid = 12.0
    chi_2_raster = np.zeros(len(phi0_raster))
    chi_3_raster = np.zeros(len(phi0_raster))
    chi_4_raster = np.zeros(len(phi0_raster))
    mu_raster = np.zeros(len(phi0_raster))

    rho_array = np.zeros(len(phi0_raster))
    mu_array = np.zeros(len(phi0_raster))

    for i in range(0, len(phi0_raster)):
        phi0 = phi0_raster[i]

        rho_array[i] = rho_Phi1r(Phi1r, phi0, V_args, f_args, r_mid)
        mu_array[i] = mu_Phi1r(Phi1r, phi0, V_args, f_args, r_mid)


        chi_2_raster[i] = chi_i_calc_musmall(Phi1r, phi0, V_args, f_args, r_mid, 2, d_x, npts)
        if math.isnan(chi_2_raster[i])==1:
            print '\nproblematic point: ', i, Phi1r
            continue
        chi_3_raster[i] = chi_i_calc_musmall(Phi1r, phi0, V_args, f_args, r_mid, 3, d_x, npts)
        #chi_3_raster[i] = chi_3_from_chi2(Phi1r, phi0, V_args, f_args, r_mid, d_x, npts)
        chi_4_raster[i] = chi_i_calc_musmall(Phi1r, phi0, V_args, f_args, r_mid, 4, d_x, npts)
        #chi_4_raster[i] = chi_4_from_chi3(Phi1r, phi0, V_args, f_args, r_mid, d_x, npts)
        #chi_5_raster[i] = chi_5_from_chi4(Phi1r, phi0, V_args, f_args, r_mid, d_x, npts)
        #chi_6_raster[i] = chi_6_from_chi5(Phi1r, phi0, V_args, f_args, r_mid, d_x, npts)

        TD_pts.clear()
        tdp0.clear()

        TDc_pw = TD_calc_pointwise(phi0, Phi1r, V_args, f_args, r_mid)
        r_mid = TDc_pw[2]
        TD_pt = TDc_pw[0]
        TD_pt_scaled = TD_pt*lambdas
        mu_raster[i] = TD_pt_scaled[2]
        print 20*'#'
        print 'T = %2.6f, mu = %2.6f' %(TD_pt_scaled[0], TD_pt_scaled[2])
        print 'i =', i, 'phi_0 =', phi0_raster[i], 'Phi_1r =', Phi1r
        print 'chi_2 = %2.6f, chi_3 = %2.6f, chi_4 = %2.6f' %(chi_2_raster[i], chi_3_raster[i], chi_4_raster[i])
        print 20*'#'



    ## Plots:
    figure(1)   # chi2/T2
    plot(T_raster, chi_2_raster/T_raster**2.0*lambdas[3]/lambdas[2], label = r'$\mu = %2.4f \, [MeV]$' %TD_pt_scaled[2])
    xlabel(r'$T \, [MeV]$')
    ylabel(r'$\chi_2/T^{\,2}$')
    legend(frameon = False, loc = 'best')

    figure(2)   # chi3/T
    plot(T_raster, chi_3_raster/T_raster*lambdas[3]/lambdas[2]**2.0, label = r'$\mu = %2.4f \, [MeV]$' %TD_pt_scaled[2])
    xlabel(r'$T \, [MeV]$')
    ylabel(r'$\chi_3/T$')
    legend(frameon = False, loc = 'best')

    figure(3)   # chi4
    plot(T_raster, chi_4_raster*lambdas[3]/lambdas[2]**3.0, label = r'$\mu = %2.4f \, [MeV]$' %TD_pt_scaled[2])
    xlabel(r'$T \, [MeV]$')
    ylabel(r'$\chi_4$')
    legend(frameon = False, loc = 'best')

    figure(4)
    plot(T_raster, mu_raster)
    # figure(4)
    # plot(T_raster, chi_5_raster*T_raster*lambdas[3]/lambdas[2]*lambdas[0])
    # #plot(T_raster, chi_4_raster_2)
    # xlabel(r'$T \, [MeV]$')
    # ylabel(r'$\chi_5 T$')
    # figure(5)
    # plot(T_raster, chi_6_raster*T_raster**2.0*lambdas[3]/lambdas[2]*lambdas[0]**2.0)
    # #plot(T_raster, chi_4_raster_2)
    # xlabel(r'$T \, [MeV]$')
    # ylabel(r'$\chi_6 T{\,2}$')

    print 'chi_2_raster: ', chi_2_raster
    print 'chi_3_raster: ', chi_3_raster

    chis_dic.update({Phi1r:[chi_2_raster, chi_3_raster, chi_4_raster, TD_pt_scaled[2]]})






#### brute force:
#figure(5)
#rho_array, mu_array, T_raster = zip(*sorted(zip(rho_array, mu_array, T_raster)))
#plot(rho_array, mu_array)
#rhomu_tck = splrep(rho_array, mu_array)
#chi_der = 1.0/splev(rho_array, rhomu_tck, der=1)
#figure(6)
#plot(T_raster,chi_der)
####



for key in chis_dic.keys():
    print key, chis_dic[key][1]


fname = model_type+'/'+ftype+'/chis_list_'+model_type+'2.p'
pickle.dump(chis_dic, open(fname, "wb"))
file.close(open(fname))


show()


print 'done: chis_musmall_numcalc2.py'