import numpy as np
from scipy.integrate import quad, odeint
from scipy.interpolate import splrep, splev
from amoba import amoeba
from args_and_lambds import args_dic, lambdas_dic
from ftypes import fs, dfs

def V(phi, *args):
    return Vs[args[0]](phi, *args[1])

def dV_dphi(phi, *args):
    return dVs[args[0]](phi, *args[1])

def f(phi, *args):
    return fs[args[0]](phi, *args[1])

def df_dphi(phi, *args):
    return dfs[args[0]](phi, *args[1])

from Vtypes import Vs, dVs, dlogVs
from H_expansions_G import get_horizon_expansions
from args_and_lambds import args_dic, lambdas_dic

from rasterizer import rasterize
from time import time
import pickle

from backbone_g_5_mu0calc2_asmodule import chi2_mu0_calc_pt

from pylab import figure, plot, legend, show, axvline

def chi2_fit(p, data):
    metric_data = data[0]
    TD_data = data[1]
    phi0_raster = data[2]
    lat_data = data[3]
    lambdas = data[4]
    ftype = data[5]

    f_args = [ftype, list(p)]
    chi2T2_scale = data[6]
    #f_args = [ftype, list(p[:-1])]
    #chi2T2_scale = p[-1]
    print f_args, chi2T2_scale
    chi2T2_raster = np.zeros(len(phi0_raster))
    for i in range(0, len(phi0_raster)):
        TD_point = TD_data[i, :]
        metric_sol = metric_data[i]
        chi2T2_raster[i] = chi2T2_scale*chi2_mu0_calc_pt(TD_point, metric_sol, f_args, i, 0)/f(0, *f_args)

    T_lat = lat_data[:, 0]
    chi2T2_lat = lat_data[:, 1]
    T_holo = TD_data[:, 0]
    #chi2T2_tck = splrep(T_holo[::-1], chi2T2_raster[::-1])
    chi2T2_tck = splrep(T_holo, chi2T2_raster)
    chi2T2_model = splev(T_lat, chi2T2_tck)
    print chi2T2_model
    chiq = np.sum((chi2T2_lat - chi2T2_model)**2.0)/np.float(len(phi0_raster))
    print 'chiq =', chiq
    chiq = np.log(chiq)
    print '- log chiq =', - chiq

    return - chiq

def fit_f(p, data):
    f_raster_init = data[0]
    f_type = data[1]
    phi0_raster = data[2] ##cut off to relevant region

    p_f = [f_type, np.array(p)]
    print p_f
    f_raster_new = rasterize(f, phi0_raster, *p_f)[1]
    chiq = np.sum((f_raster_new - f_raster_init)**2.0)/np.float(len(phi0_raster))
    print chiq
    chiq = np.log(chiq)
    print '- log chiq =', - chiq
    return - chiq

def compress_TDTA_to_lat_range(phi0_min, phi0_max, phi0_raster, TD_data_mu0, metric_data_mu0):
    ind_1 = len(np.compress(phi0_raster <= phi0_min, phi0_raster))
    ind_2 = len(phi0_raster) - len(np.compress(phi0_raster >= phi0_max, phi0_raster))
    phi0_raster_new = phi0_raster[ind_1:ind_2 ]
    TD_data_mu0_new = TD_data_mu0[ind_1:ind_2, :]
    metric_data_mu0_new = metric_data_mu0[ind_1:ind_2]

    print phi0_raster_new
    print TD_data_mu0_new

    return [phi0_raster_new, TD_data_mu0_new, metric_data_mu0_new]

suffx = 'no'
metric_data_mu0 = pickle.load(open('metric_data_'+suffx+'.p', "rb"))
file.close(open('metric_data_'+suffx+'.p'))
TDTA_mu0 = pickle.load(open('TDTA_'+suffx+'.p', "rb"))
file.close(open('TDTA_'+suffx+'.p'))
TD_data_mu0 = TDTA_mu0[0]
phi0_raster = TDTA_mu0[1]
V_args = args_dic['V'][suffx] # V_args = TDTA_mu0[-1]

lat_data = pickle.load(open('chi2_wubp.p', "rb"))
file.close(open('chi2_wubp.p'))

model_type = 'no'
#f_pars_init = args_dic['f'][model_type]
lambdas_init = lambdas_dic[model_type]
#chi2scale_init = lambdas_init[3]/lambdas_init[2]/lambdas_init[0]**2.0
chi2scale_init = lambdas_init[3]*lambdas_init[0]/(lambdas_init[1]*lambdas_init[2])#*0.28/0.35
print chi2scale_init
if model_type == 'no':
    phi0_raster_ff = np.compress(phi0_raster < 3.4, phi0_raster)
    phi0_raster_ff = np.compress(phi0_raster_ff > 0.52, phi0_raster_ff)
if suffx == 'G':
    phi0_raster_ff = np.compress(phi0_raster < 7.0, phi0_raster)
    phi0_raster_ff = np.compress(phi0_raster_ff > 1.8, phi0_raster_ff)
elif suffx == 'VRY_1':
    phi0_raster_ff = np.compress(phi0_raster < 4.2, phi0_raster)
    phi0_raster_ff = np.compress(phi0_raster_ff > 0.4, phi0_raster_ff)


############## best fit w Gubser V ##################
# ftype = 'f_cq_thsa'
# #x0 = np.array([8.6782, 1.2162, 5.2326, 3.2981, -1.1880]) ## f_cq_thsa log chiq = 12.6569 bbbbest fit <========
# #x0 = np.array([8.6776, 1.2161, 5.2327, 3.2977, -1.1885]) ## f_cq_thsa
# x0 = np.array([0, 1.9161, 2.5327, 2.8977, -1.5885, chi2scale_init]) ## f_cq_thsa
# ############### best fit ##################
# scale = x0/40.0


ftype = 'f_no'
ftype = 'f_tanh'


### old scale
chi2scale_init = 0.5
#x0 = np.array([2.8977, -1.5885, 1.9161, 2.1327, chi2scale_init]) #chiq = 9.9169
x0 = np.array([1.0, -6.5, 0.4, 5.0, chi2scale_init]) #chiq = 13.6036 w VRY_1
#chi2scale_init = 0.05517
#x0 = np.array([2.93377, -2.57227, 1.56084, 2.36919, chi2scale_init]) #chiq = 13.313 w VRY_2
#chi2scale_init = 0.05537
#x0 = np.array([2.92335, -2.56313, 1.56084, 2.36919, chi2scale_init]) #chiq = 13.313 w VRY_2
#x0 = np.array([1.0/3.0*np.cosh(0.69), 1.2, 0.69/1.2, 2.0/3.0, -100.0, chi2scale_init]) #
scale = x0

##### proper scaling of T and s
#chi2scale_init = 0.06449
#x0 = np.array([2.99800, -2.65038, 1.54966, 2.18202, chi2scale_init]) #chiq = 13.77 w VRY_2
#scale = x0/40.0
#####
#chi2scale_init = 1.0
#x0 = np.array([2.99800*0.06449, -2.65038*0.06449, 1.54966, 2.18202])
#f0 = f(0, *['f_tanh', x0])
#print 'f(0) =', f0
#chi2scale_init = f0
#scale = x0/20.0
###1st and 2nd pars can be obtained from previous fit via p_1new = p_1old*chi2scale_old, p_2new = p_2old*chi2scale_old


print 'model type =', model_type
print 'x0 =', x0
print 'scale =', scale

print len(phi0_raster)
compressed_mdata = compress_TDTA_to_lat_range(0.2, 4.2, phi0_raster, TD_data_mu0, metric_data_mu0)
phi0_raster = compressed_mdata[0][::-1]
TD_data_mu0 = compressed_mdata[1][::-1]
metric_data_mu0 = compressed_mdata[2][::-1]
print 'buh'
print phi0_raster
print lat_data[:,0]
print TD_data_mu0[:,0]
print len(phi0_raster)

f_args_init = [ftype, list(x0)]
chi2_init_raster = np.zeros(len(phi0_raster))
for i in range(0, len(phi0_raster)):
    chi2_init_raster[i] = chi2_mu0_calc_pt(TD_data_mu0[i, :], metric_data_mu0[i], f_args_init, 0, 0)/f(0, *f_args_init)

ind_1 = len(np.compress(TD_data_mu0[:,0] <= lat_data[:,0][0], TD_data_mu0[:,0]))
ind_2 = len(TD_data_mu0[:,0]) - len(np.compress(TD_data_mu0[:,0] >= lat_data[:,0][-1], TD_data_mu0[:,0]))
print ind_1, ind_2

figure(1)
plot(TD_data_mu0[:, 0], chi2_init_raster*chi2scale_init, label = r'$\chi_2/T^2$ model')
plot(lat_data[:, 0], lat_data[:, 1], label = r'$\chi_2/T^2$ lattice')


figure(2)
print V_args
V_raster = rasterize(V, phi0_raster, *V_args)[1]
dV_raster = rasterize(dV_dphi, phi0_raster, *V_args)[1]

plot(phi0_raster, dV_raster/V_raster, label = r'$V^\prime/V$')
plot(phi0_raster, rasterize(f, phi0_raster, *f_args_init)[1], label = r'$f$')
axvline(x = phi0_raster[ind_1])
axvline(x = phi0_raster[ind_2])
legend()

figure(1)
plot(TD_data_mu0[:, 0], dV_raster/V_raster, label = r'$V^\prime/V$')
plot(TD_data_mu0[:, 0], rasterize(f, phi0_raster, *f_args_init)[1], label = r'$f$')
legend()


figure(3)
plot(phi0_raster, TD_data_mu0[:, 0])
plot(phi0_raster, chi2_init_raster*chi2scale_init*1000.0)
axvline(x = phi0_raster[ind_1])
axvline(x = phi0_raster[ind_2])

#print TD_data_mu0


show()

f_pars_fitted = amoeba(list(x0), list(scale), chi2_fit, xtolerance=1e-5, ftolerance=1e-7, data = ([metric_data_mu0, TD_data_mu0, phi0_raster, lat_data, lambdas_init, ftype, chi2scale_init]))
#f_pars_fitted = amoeba(list(x0), list(scale), chi2_fit, xtolerance=1e-5, ftolerance=1e-7, data = ([metric_data_mu0, TD_data_mu0, phi0_raster, lat_data, lambdas_init, ftype]))

print f_pars_fitted