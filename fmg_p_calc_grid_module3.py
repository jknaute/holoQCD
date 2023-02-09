import numpy as np
from scipy.integrate import quad
from scipy.interpolate import splrep, splev
import pickle

from p_calcer4 import p_calc_line
from rasterizer import rasterize

from args_and_lambds import args_dic, lambdas_dic
from fmg_TDprocess import TD_scale

from pylab import figure, plot, show, legend

model_type = 'VRY_2'
ftype = args_dic['ftype'][model_type]
V_args = args_dic['V'][model_type]
f_args = args_dic['f'][model_type]
lambdas = lambdas_dic[model_type]

if model_type == 'G':
    fname = 'TD_gr_G_wmu0.p'
    #fname2 = 'TD_phiPhi.p'
if model_type == 'no':
    fname = 'TD_gr_no.p'
if model_type == 'VRY_2':
    fname = model_type+'/'+ftype+'/TD_gr_VRY_2_wmu0.p'
    fname2 = model_type+'/'+ftype+'/TDTA_VRY_2.p'

print fname
TD_gr = pickle.load(open(fname, "rb"))
file.close(open(fname))
TDTA = pickle.load(open(fname2, "rb"))
file.close(open(fname2))

TD_grid = TD_gr[0]
TD_grid = TD_scale(TD_grid, lambdas)[0]
phiPhi_grid = TD_gr[1]
TD_full = TD_gr[2]

def p_integrand(Phi1r, tcks_list):
    T_tck = tcks_list[0]
    logs_tck = tcks_list[1]
    mu_tck = tcks_list[2]
    n_tck = tcks_list[3]

    return np.exp(splev(Phi1r, logs_tck))*splev(Phi1r, T_tck, der = 1) + splev(Phi1r, n_tck)*splev(Phi1r, mu_tck, der = 1)

def p_init_calc_on_Tax(TD_grid, phiPhi_grid, TDTA):
    ### calculate pressure on T-axis
    #first calculate accurate pressure from TDTA and phi0 ~= 10
    TD_Tax = TDTA[0]
    phi0_raster = TDTA[1]
    Phiphi_T0 = np.vstack((np.zeros(len(phi0_raster)), phi0_raster))
    p_Tax = p_calc_line([TD_Tax[0,0], 0], [TD_Tax[-1,0],0], TD_Tax, Phiphi_T0)
    print 'p on T axis from TDTA:', p_Tax
    pT4_Tax_tck = splrep(p_Tax[1][::-1], p_Tax[0][::-1]/p_Tax[1][::-1]**4.0)
    plot(p_Tax[1][::-1], p_Tax[0][::-1]/p_Tax[1][::-1]**4.0)

    TD_grid_Tax = TD_grid[:, 0, :]
    Phiphi_Tax = np.transpose(phiPhi_grid[:, 0][:, [1, 0]])
    p_Tax_grid = p_calc_line([TD_grid_Tax[0, 0], 0],[TD_grid_Tax[-1, 0], 0], TD_grid_Tax, Phiphi_Tax)
    print 'p on T axis from grid TD:', p_Tax_grid
    plot(p_Tax_grid[1][::-1], p_Tax_grid[0][::-1]/p_Tax_grid[1][::-1]**4.0)

    T_init_gr = p_Tax_grid[1][-1]
    p_init_true = splev(T_init_gr, pT4_Tax_tck)*T_init_gr**4.0
    print 'p init true: ', p_init_true

    p_grid_true = p_Tax_grid[0] - p_Tax_grid[0][-1] + p_init_true
    print 'p_grid_true: ', p_grid_true
    plot(p_Tax_grid[1], p_grid_true/p_Tax_grid[1]**4.0)

    show()

    return p_grid_true

def p_calc_grid(TD_grid, phiPhi_grid, Phirp_init, saveopts):
    print 'calculating p on TD grid...'
    p_grid = np.zeros((len(phiPhi_grid[:, 0][:, 0]), len(phiPhi_grid[0, :][:, 1])))
    print 'dim p_array: ', len(phiPhi_grid[:, 0][:, 0]), len(phiPhi_grid[0, :][:, 1])
    Phi1r_inits = Phirp_init[0]
    p_inits = Phirp_init[1]
    print 'Phirp_init: ', Phirp_init
    print 'phiPhi_grid[:, 0]: ', phiPhi_grid[:, 0], len(phiPhi_grid[:, 0])


    for i in range(0, len(phiPhi_grid[:, 0])):
        print i, phiPhi_grid[i, 0]
        Phi1r_raster = phiPhi_grid[i, :][:, 1]
        TD_slice_Phi1r = TD_grid[i, :, :]
        print 'TD_slice_Phi1r: ', TD_slice_Phi1r

        useinds = np.where(TD_slice_Phi1r[:, 2] < 5.0*1e4)[0] # mu < 5*1e4
        useinds = np.where(TD_slice_Phi1r[:, 0][useinds] > 0)[0] # T > 0

        Phi1r_raster_u = Phi1r_raster[useinds]
        T_raster_Phi1r_u = TD_slice_Phi1r[:, 0][useinds]
        logs_raster_Phi1r_u = np.log(TD_slice_Phi1r[:, 1][useinds])
        mu_raster_Phi1r_u = TD_slice_Phi1r[:, 2][useinds]
        n_raster_Phi1r_u = TD_slice_Phi1r[:, 3][useinds]

        T_Phi1r_u_tck = splrep(Phi1r_raster_u, T_raster_Phi1r_u)
        logs_Phi1r_u_tck = splrep(Phi1r_raster_u, logs_raster_Phi1r_u)
        mu_Phi1r_u_tck = splrep(Phi1r_raster_u, mu_raster_Phi1r_u)
        n_Phi1r_u_tck = splrep(Phi1r_raster_u, n_raster_Phi1r_u)
        tck_list = [T_Phi1r_u_tck, logs_Phi1r_u_tck, mu_Phi1r_u_tck, n_Phi1r_u_tck]

        #plot(Phi1r_raster_u, T_raster_Phi1r_u, label='T')
        #plot(Phi1r_raster_u, np.exp(logs_raster_Phi1r_u)/T_raster_Phi1r_u**3.0, label='sT3')
        #plot(Phi1r_raster_u, mu_raster_Phi1r_u, label='mu')
        #plot(Phi1r_raster_u, n_raster_Phi1r_u/T_raster_Phi1r_u**3.0, label='nT3')
        #legend()
        #show()

        integrand_raster = rasterize(p_integrand, Phi1r_raster_u, tck_list)[1]
        integrand_raster2 = rasterize(p_integrand, np.linspace(Phi1r_raster_u[0], Phi1r_raster_u[-1], 400), tck_list)[1]
        figure(1)
        plot(Phi1r_raster_u, integrand_raster)
        plot(np.linspace(Phi1r_raster_u[0], Phi1r_raster_u[-1], 400), integrand_raster2, ls = 'dashed')

        restinds1 = np.where(TD_slice_Phi1r[:, 2] > 5.0*1e4)[0]
        restinds2 = np.where(TD_slice_Phi1r[:, 0] == 0)[0]
        restinds = np.hstack((restinds1, restinds2))
        print 'useinds:', useinds
        print 'restinds:', restinds
        print 'Phi1r_raster:', Phi1r_raster_u
        print 'initial vals: Phi_1r =', Phi1r_inits[i], ' p =', p_inits[i]

        for j in range(0, len(Phi1r_raster)):
            if j in useinds:
                p_grid[i, j] = p_inits[i] + quad(p_integrand, Phi1r_inits[i], Phi1r_raster[j], args = tck_list, epsabs = 1e-9, epsrel = 1e-10, limit = 1000)[0]
            elif j in restinds:
                p_grid[i, j] = 0

        print 'T:', TD_slice_Phi1r[:, 0], 'mu:', TD_slice_Phi1r[:, 2]
        print 'pT4:', p_grid[i, :]/TD_slice_Phi1r[:, 0]**4.0

    print '...finished calculating p on TD grid'
    if saveopts[0]:
        savename = saveopts[1]
        print 'saving to '+savename
        pickle.dump(p_grid, open(savename, "wb"))
        file.close(open(savename))

    show()
    return p_grid

#print phiPhi_grid[0, :]

saveopts = [1, model_type+'/'+ftype+'/p_grid_VRY_2.p']

p_Tax_grid = p_init_calc_on_Tax(TD_grid, phiPhi_grid, TDTA)
print 'p_Tax =', p_Tax_grid
print 'phiPhi_grid[:, 0][:, 1]: ', phiPhi_grid[:, 0][:, 1]
Phip_Tax = [phiPhi_grid[:, 0][:, 1], p_Tax_grid]
p_calc_grid(TD_grid, phiPhi_grid, Phip_Tax, saveopts)