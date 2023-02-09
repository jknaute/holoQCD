import numpy as np
import numpy.ma as ma
from fmg_TDprocess import TD_scale, TD_scale_isen, J_calc_fd
from scipy.interpolate import splrep, splev
from scipy.optimize import fmin_l_bfgs_b, brentq
from amoba import amoeba

from pylab import * # figure, plot, legend, show, semilogy, scatter, xlabel, ylabel, rc, axis, savefig, subplot, contour, contourf, colorbar, cm, axhline, axvline, getp#, add_axes

from TD_calc_pow import TD_calc_pointwise
from args_and_lambds import args_dic, lambdas_dic
from HEE_calc import hee_integration

import pickle
from matplotlib.transforms import Bbox
from nice_tcks import nice_ticks


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



def get_contour_verts(cn, xy_rev, path_rev):
    contours = []
    # for each contour line
    for cc in cn.collections:
        paths = []
        # for each separate section of the contour line
        for pp in cc.get_paths():
            xy = []
            # for each segment of that section
            for vv in pp.iter_segments():
                if xy_rev:
                    xy.append(vv[0][::-1])
                else:
                    xy.append(vv[0])
            if path_rev:
                paths.append(np.vstack(xy)[::-1])
            else:
                paths.append(np.vstack(xy))
        contours.append(paths)

    return contours

def get_contours(cnt, xy_rev, path_rev, levels, rem_lr, T_or_mu):
    cont_list = [None]*len(levels)
    cont_dic = {}
    gcv = get_contour_verts(cnt, xy_rev, path_rev)
    #print gcv
    if T_or_mu == 'T':
        exind = 1
    elif T_or_mu == 'mu':
        exind = 0
    for k in range(0, len(levels)):
        seg_num = len(gcv[k])
        print seg_num, levels[k]
        if rem_lr:
            if seg_num == 2:
                cont_dic.update({levels[k]:gcv[k][exind]})
                cont_list[k] = gcv[k][exind]
            elif seg_num == 1:
                cont_dic.update({levels[k]:gcv[k]})
                cont_list[k] = gcv[k]
            elif seg_num == 3:
                cont_dic.update({levels[k]:[gcv[k][0], gcv[k][2]]})
                cont_list[k] = [gcv[k][0], gcv[k][2]]
        else:
            cont_dic.update({levels[k]:gcv[k][0]})
            cont_list[k] = gcv[k][0]
        print cont_dic
        print levels[k], 'index length of contour', len(cont_dic[levels[k]])

    return [cont_list, cont_dic]

def get_more_T_contourpoints(num_pts, cntr_approx, derv_cutoff):
    lphi0_appr = cntr_approx[:, 0]
    print 'lphi0_appr: ', lphi0_appr
    Phi1r_appr = cntr_approx[:, 1]
    print 'Phi1r_appr: ', Phi1r_appr
    npo = len(lphi0_appr)
    print 'npo: ', npo
    derv_raster = np.zeros(npo)
    fitwhat = np.zeros((2, num_pts))

    ##calculate finite difference derivative and check if T contour exhibits reverse-s shape or not
    for i in range(0, npo - 1):
        i_sw1 = npo - 1
        derv_raster[i] = (lphi0_appr[i + 1] - lphi0_appr[i])/(Phi1r_appr[i + 1] - Phi1r_appr[i])
    derv_raster[-1] = (lphi0_appr[-1] - lphi0_appr[-2])/(Phi1r_appr[-1] - Phi1r_appr[-2])
    print 'derv_raster: ', derv_raster
    print 'min/max deriv:', np.amin(derv_raster), np.amax(np.abs(derv_raster))
    print 'np.amax(np.abs(derv_raster)): ', np.amax(np.abs(derv_raster))

    if np.amax(np.abs(derv_raster)) < 30: ##no reverse s-shape
        Phi1r_new1 = np.linspace(Phi1r_appr[0]**0.2, 0.0005**0.2, num_pts/3)**5.0
        #Phi1r_new2 = np.linspace(0.000501, Phi1r_appr[-1], num_pts - 30)
        Phi1r_new2 = np.linspace(0.000505**0.5, Phi1r_appr[-1]**0.5, num_pts - (num_pts/3))**2.0

        Phi1r_new = np.hstack((Phi1r_new1, Phi1r_new2))
        #plot(Phi1r_appr, lphi0_appr)
        #show()
        Phi1r_appr, lphi0_appr = zip(*sorted(zip(Phi1r_appr, lphi0_appr)))
        cntr_tck = splrep(Phi1r_appr, lphi0_appr, k = 1) ##no rss => no probs with spline interpolation
        lphi0_new = splev(Phi1r_new, cntr_tck)
        fitwhat[1, :] = np.ones(num_pts) ##keep Phi1r fixed and fit lphi0

    else: ##reverse s-shape
        for i in range(0, npo):
            if derv_raster[i] < - derv_cutoff:
                print i, '<', derv_raster[i]
                i_sw1 = i
                break
        for i in range(i_sw1, npo):
            if derv_raster[i] < 0 and derv_raster[i] > - derv_cutoff:
                print i, '>', derv_raster[i]
                i_sw2 = i
                break
        print i_sw1, i_sw2
        print lphi0_appr[i_sw1], lphi0_appr[i_sw2]

        # ind_0 = 10
        ind_1 = num_pts/2
        ind_2 = num_pts/3
        ind_3 = num_pts/6
        ##ind_1 + ind_2 + ind_3 must be = num_pts

        Phi1r_new1 = np.linspace(Phi1r_appr[0], Phi1r_appr[i_sw1 - 1], ind_1)
        #Phi1r_new_1 = np.linspace(Phi1r_appr[0]**0.2, 0.0005**0.2, 30)**5.0
        #Phi1r_new_2 = np.linspace(0.000501, Phi1r_appr[i_sw1 - 1], ind_1 - 30)
        #Phi1r_new_2 = np.linspace(0.000505**0.5, Phi1r_appr[i_sw1 - 1]**0.5, ind_1 - 30)**2.0
        #Phi1r_new1 = np.hstack((Phi1r_new_1, Phi1r_new_2))

        #plot(Phi1r_appr[:i_sw1], lphi0_appr[:i_sw1])
        #show()
        cntr_tck1 = splrep(Phi1r_appr[:i_sw1], lphi0_appr[:i_sw1]) ##no rss => no probs with spline interpolation
        lphi0_new1 = splev(Phi1r_new1, cntr_tck1)
        fitwhat[1, :ind_1] = np.ones(ind_1)

        lphi0_new2 = np.linspace(lphi0_appr[i_sw1], lphi0_appr[i_sw2 - 1], ind_2)
        #plot(lphi0_appr[i_sw1:i_sw2][::-1], Phi1r_appr[i_sw1:i_sw2][::-1])
        #show()
        cntr_tck2 = splrep(lphi0_appr[i_sw1:i_sw2][::-1], Phi1r_appr[i_sw1:i_sw2][::-1])
        Phi1r_new2 = splev(lphi0_new2, cntr_tck2)
        fitwhat[0, ind_1:ind_1 + ind_2] = np.ones(ind_2)

        Phi1r_new3 = np.linspace(Phi1r_appr[i_sw2], Phi1r_appr[-1], ind_3)
        #plot(Phi1r_appr[i_sw2:], np.log(lphi0_appr[i_sw2:]))
        #show()
        cntr_tck3 = splrep(Phi1r_appr[i_sw2:], np.log(lphi0_appr[i_sw2:]), k = 1) ##no rss => no probs with spline
        # interpolation
        lphi0_new3 = np.exp(splev(Phi1r_new3, cntr_tck3))
        fitwhat[1, ind_1 + ind_2:] = np.ones(ind_3)

        Phi1r_new = np.hstack((Phi1r_new1, Phi1r_new2, Phi1r_new3))
        lphi0_new = np.hstack((lphi0_new1, lphi0_new2, lphi0_new3))

    return [Phi1r_new, lphi0_new, fitwhat]

def get_more_mu_contourpoints(num_pts, cntr_approx, derv_cutoff):
    if len(cntr_approx) != 2:
        lphi0_appr = cntr_approx[:, 0]
        Phi1r_appr = cntr_approx[:, 1]
        print lphi0_appr, Phi1r_appr
        fitwhat = np.zeros((2, num_pts))

        if np.amax(Phi1r_appr) < 0.8: ##simplest case -> take lphi0 as good monotonous coord
            lphi0_new = np.linspace(lphi0_appr[0], lphi0_appr[- 1], num_pts)
            cntr_tck = splrep(lphi0_appr, Phi1r_appr, k = 3)
            Phi1r_new = splev(lphi0_new, cntr_tck)
            fitwhat[0, :] = np.ones(num_pts)

        elif np.amax(Phi1r_appr) >= 0.8:
            print 'buh'
            Phi1r_argmax = np.argmax(Phi1r_appr)

            lphi0_appr_l = lphi0_appr[:Phi1r_argmax]
            Phi1r_appr_l = Phi1r_appr[:Phi1r_argmax]
            fitwhat_l = np.zeros((2, num_pts/2.0))

            lphi0_appr_u = lphi0_appr[Phi1r_argmax:]
            Phi1r_appr_u = Phi1r_appr[Phi1r_argmax:]
            fitwhat_u = np.zeros((2, num_pts/2.0))

            Phi1r_new_l = np.linspace(Phi1r_appr_l[0], Phi1r_appr_l[-1], int(num_pts/2.0))
            Phi1r_appr_l, lphi0_appr_l = zip(*sorted(zip(Phi1r_appr_l, lphi0_appr_l))) # sort arrays
            cntr_l_tck = splrep(Phi1r_appr_l, lphi0_appr_l, k = 3)
            lphi0_new_l = splev(Phi1r_new_l, cntr_l_tck)
            fitwhat_l[1, :] = np.ones(int(num_pts/2.0))

            Phi1r_new_u = np.linspace(Phi1r_appr_u[0], Phi1r_appr_u[-1], int(num_pts/2.0))
            Phi1r_appr_u, lphi0_appr_u = zip(*sorted(zip(Phi1r_appr_u, lphi0_appr_u)))
            cntr_u_tck = splrep(Phi1r_appr_u, lphi0_appr_u, k = 3)
            #cntr_u_tck = splrep(Phi1r_appr_u[::-1], lphi0_appr_u[::-1], k = 3)
            lphi0_new_u = splev(Phi1r_new_u, cntr_u_tck)
            fitwhat_u[1, :] = np.ones(int(num_pts/2.0))

            #print fitwhat_l, fitwhat_u
            Phi1r_new = np.hstack((Phi1r_new_l, Phi1r_new_u))
            lphi0_new = np.hstack((lphi0_new_l, lphi0_new_u))

            fitwhat = np.vstack((np.hstack((fitwhat_l[0, :], fitwhat_u[0, :])), np.hstack((fitwhat_l[1, :], fitwhat_u[1, :]))))
            #print fitwhat_new

        return [Phi1r_new, lphi0_new, fitwhat]

    elif len(cntr_approx) == 2:
        lphi0_appr_l = cntr_approx[0][:, 0]
        Phi1r_appr_l = cntr_approx[0][:, 1]
        fitwhat_l = np.zeros((2, num_pts/2.0))

        lphi0_appr_u = cntr_approx[1][:, 0]
        Phi1r_appr_u = cntr_approx[1][:, 1]
        fitwhat_u = np.zeros((2, num_pts/2.0))

        #print Phi1r_appr_l, lphi0_appr_l, Phi1r_appr_u, lphi0_appr_u

        Phi1r_new_l = np.linspace(Phi1r_appr_l[0], Phi1r_appr_l[-1], int(num_pts/2.0))
        cntr_l_tck = splrep(Phi1r_appr_l, lphi0_appr_l, k = 3)
        lphi0_new_l = splev(Phi1r_new_l, cntr_l_tck)
        fitwhat_l[1, :] = np.ones(int(num_pts/2.0))

        Phi1r_new_u = np.linspace(Phi1r_appr_u[0], Phi1r_appr_u[-1], int(num_pts/2.0))
        cntr_u_tck = splrep(Phi1r_appr_u[::-1], lphi0_appr_u[::-1], k = 3)
        lphi0_new_u = splev(Phi1r_new_u, cntr_u_tck)
        fitwhat_u[1, :] = np.ones(int(num_pts/2.0))

        Phi1r_new = np.hstack((Phi1r_new_l, Phi1r_new_u))
        lphi0_new = np.hstack((lphi0_new_l, lphi0_new_u))

        fitwhat = np.vstack((np.hstack((fitwhat_l[0, :], fitwhat_u[0, :])), np.hstack((fitwhat_l[1, :], fitwhat_u[1, :]))))

        return [Phi1r_new, lphi0_new, fitwhat]

def phi_or_Phi_fit_b(fit_fld, fixfld, fitwhat, TD_val, whichfunc_ind, V_args, f_args, r_mid):
    phi0 = np.array([fit_fld, fixfld])[fitwhat.argmin()]
    Phi1 = np.dot(np.array([fit_fld, fixfld]), fitwhat)
    print 'fix field:', fixfld, 'fit field:', fit_fld, 'phi0:', phi0, 'Phi1:', Phi1, whichfunc_ind

    TD_pt = TD_calc_pointwise(phi0, Phi1, V_args, f_args, r_mid)
    func_val = TD_pt[0][whichfunc_ind]*lambdas[whichfunc_ind]
    print 4*'#'
    print 'phi_0 = %2.12f, Phi_1r = %2.12f, func_val = %2.8f, TD_val = %2.8f' %(phi0, Phi1, func_val, TD_val) #'TD_pt[0][whichfunc_ind]
    #print 'TD_all:', TD_pt[0]*lambdas
    print 'T = %2.5f, s = %2.5f, mu = %2.5f, rho = %2.5f' %tuple([TD_pt[0][k]*lambdas[k] for k in range(0, 4)])

    print 4*'#'
    #chiq = - np.log((func_val - TD_val)**2.0)
    diff = func_val - TD_val
    print 'diff =', diff

    return diff

def get_acc_contour_and_TD(contour, cnt_val, TDfunc, V_args, f_args):
    Phi1r_raster = contour[0]
    lphi0_raster = contour[1]
    fitwhat = contour[2]
    phi0_raster = np.exp(lphi0_raster)

    Phiphi_acc = np.zeros((2, len(Phi1r_raster)))
    TD_slice = np.zeros((len(Phi1r_raster), 4))
    metric_dic_lvl = {}

    #print Phiphi_acc[:, 0]
    #print fitwhat[:, 0]
    TD_inds = {'T':0, 's':1, 'mu':2, 'rho':3}
    fields_raster = np.vstack((Phi1r_raster, phi0_raster))
    if TDfunc == 'mu':
        r_mid = 12.0
    elif TDfunc == 'T':
        if cnt_val <= 150.0:
            r_mid = 2.0
        else:
            r_mid = 8.0

    for i in range(0, len(Phi1r_raster)):
        if cnt_val > 0:
            fields = np.array([Phi1r_raster[i], phi0_raster[i]])
            fix_field = fields[fitwhat[:, i].argmin()]
            fit_field_init = np.dot(fields, fitwhat[:, i])
            print 'fields: ', fields, i, fitwhat[:, i], fix_field, fit_field_init

            Phiphi_acc[fitwhat[:, i].argmin(), i] = fix_field
            print Phiphi_acc[:, i]

            ## a and b for brentq:
            a_fac = 0.99
            b_fac = 1.01
            if i == 0 or i == len(Phi1r_raster) - 1:
                a_fac = 0.95
                b_fac = 1.05

            succ = 0
            while succ != 1: # while loop copied from ..._new.py file
                print 'succ:', succ, fit_field_init, fit_field_init*a_fac, fit_field_init*b_fac
                try:
                    print '...in try loop1...'
                    if TD_calc_pointwise(np.array([fit_field_init*a_fac, fix_field])[fitwhat[:, i].argmin()], np.dot(np.array([fit_field_init*a_fac, fix_field]), fitwhat[:, i]), V_args, f_args, r_mid) == 0: # in case there is no BH solution
                        return 0
                    fitted_field = brentq(phi_or_Phi_fit_b, fit_field_init*a_fac, fit_field_init*b_fac, xtol=1e-12, rtol=1e-10,
                                          args = (fix_field, fitwhat[:, i], cnt_val, TD_inds[TDfunc], V_args, f_args, r_mid))#[0]
                    succ = 1
                    #fit_fld, fixfld, fitwhat, TD_val, whichfunc_ind, V_args, f_args, r_mid
                except ValueError:
                    #succ = 0
                    try:
                        print '...in try loop2...'
                        phi_or_Phi_fit_b(fit_field_init*a_fac, fix_field, fitwhat[:, i], cnt_val, TD_inds[TDfunc], V_args, f_args, r_mid)
                    except ValueError:
                        fitted_field = 0
                        break
                    try:
                        print '...in try loop3...'
                        phi_or_Phi_fit_b(fit_field_init*b_fac, fix_field, fitwhat[:, i], cnt_val, TD_inds[TDfunc], V_args, f_args, r_mid)
                    except ValueError:
                        print 'beeeh'
                        fitted_field = 0
                        break
                    a_fac = a_fac*0.95
                    b_fac = b_fac*1.05

            print 10*'#'
            print fitted_field
            print 10*'#'

            phi0 = np.array([fitted_field, fix_field])[fitwhat[:, i].argmin()]
            Phi1r = np.dot(np.array([fitted_field, fix_field]), fitwhat[:, i])

        elif cnt_val == 0 and TDfunc == 'mu':
            phi0 = phi0_raster[i]
            Phi1r = 0
            fitted_field = 0

        TD_pt = TD_calc_pointwise(phi0, Phi1r, V_args, f_args, r_mid)

        if TD_pt == 0:
            return 0

        TD_slice[i, :] = TD_pt[0]
        metric_sol = TD_pt[1]
        r_mid = TD_pt[2]

        Phiphi_acc[fitwhat[:, i].argmax(), i] = fitted_field#[0]

        if save_metric or calculate_hee_lvls:
            metric_dic_lvl[i] = {'r':metric_sol['r'], 'A':metric_sol['A'], 'h':metric_sol['h']}

    print 'Phiphi_acc:', Phiphi_acc
    return [Phiphi_acc, TD_slice, metric_dic_lvl]



###############################################################################
model_type = 'VRY_4'
ftype = args_dic['ftype'][model_type]

V_args = args_dic['V'][model_type]
f_args = args_dic['f'][model_type]
lambdas = lambdas_dic[model_type]


fname = model_type+'/'+ftype+'/TD_gr_'+model_type+'_wmu0.p'
TD_gr = pickle.load(open(fname, "rb"))

TD_grid = TD_gr[0]
TD_grid = TD_scale(TD_grid, lambdas)[0]
phiPhi_grid = TD_gr[1]
TD_full = TD_gr[2]

phi0_pts = TD_full['phi0_pts']
Phi1_pts = TD_full['Phi1_pts']

lphi0_raster = np.log(TD_full['phi0_raster'])
Phi1r_raster = phiPhi_grid[20, :][:, 1]

print 'lphi0_raster: ', lphi0_raster
print 'Phi1r_raster: ', Phi1r_raster
print 'TD_grid: ', TD_grid[:, :, 0]


#### contours:
T_levels = [  50.,   75.,  100.,  125.,  150.,  175.,  200.,  225.,  250.]
T_levels = [  50.,   75.,  100.,  125.,  150.,  175.,  200.,  225.,  250., 300., 350., 400.]
T_levels = list(np.arange(55.0, 180.0, 1.0))
T_levels = [93.5, 94.5]
T_levels = [93.25, 93.75]
T_levels = [93.255]
T_levels = list(np.arange(180.0, 501.0, 10.0))
T_levels = list(np.arange(50.0, 501.0, 10.0))
T_levels1 = list(np.arange(55.0, 180.0, 1.0))
T_levels2 = list(np.arange(180.0, 501.0, 10.0))
T_levels = np.hstack((T_levels1, T_levels2))
T_levels = [93.1, 93.2, 93.22, 93.25, 93.255, 93.26, 93.28, 93.3, 93.4, 93.5, 93.75, 94.5]
T_levels = [111.7]

### for NOnew:
#T_levels = list(np.arange(70.0, 301.0, 5.0))


mu_levels = [    0.,   100.,   200.,   300.,   400.,   500.,   600.,   700.,
         800.,   850.,   875.,   900.,   910.,   920.,   930.,   940.,
         950.,   960.,   965.,   970.,   975.,   980.,   985.,   989.,
         990.,   991.,   995.,  1000.,  1005.,  1010.,  1015.,  1020.,
        1025.,  1030.,  1040.,  1045.,  1050.,  1055.,  1060.,  1065.,
        1075.,  1080.,  1090.,  1100.,  1125.,  1150.,  1175.,  1200.,
        1225.,  1250.,  1275.,  1300.,  1325.,  1350.,  1380.,  1400.,
        1420.,  1450.,  1475.,  1500.]
mu_levels = [1550., 1600., 1650., 1650., 1700.]
#mu_levels = [200., 400., 600., 800., 1000., 1010., 1020., 1030., 1040., 1050., 1075., 1100., 1125.] # mupts
#mu_levels = [900, 910, 920, 930, 940, 950, 960, 970, 980, 990] # mupts2
#mu_levels = [1150., 1175., 1200., 1250., 1300., 1350., 1400., 1450., 1500., 1550., 1600., 1650] # mupts3
#mu_levels = [0, 100, 200, 300, 500, 700, 1700] # mupts4
#mu_levels = [1380, 1400., 1420, 1450., 1500., 1550., 1600., 1650] # mupts5
#mu_levels = [850, 875, 965, 975, 985, 995, 1005, 1015, 1025, 1275, 1325] # mupts6
#mu_levels = [989, 991, 1045, 1055, 1060, 1065, 1080, 1090, 1225, 1475] # mupts7
mu_levels0 = [0.0]
mu_levels1 = [200.0, 300.0, 400.0, 500.0]
mu_levels2 = list(np.arange(600.0, 621.0, 1.0))
mu_levels3 = list(np.arange(1000.0, 1421, 20))
mu_levels4 = list(np.arange(1440.0, 1740, 20))
mu_levels5 = [987.25, 987.5, 987.75]
mu_levels = np.hstack((mu_levels2))

### for NOnew:
#mu_levels = [0.0, 400.0]

## for HEE:
mu_levels0 = np.array([0.0, 0.4, 0.8, 1.2, 1.4])*611.5
mu_levels1 = np.arange(608.0, 620.0, 1.0)
mu_levels2 = [611.0, 611.25, 611.5, 611.75, 612.0]
mu_levels3 = np.arange(620.0, 1041.0, 10)
mu_levels = np.hstack((mu_levels2))
mu_levels = list(np.arange(0, 1000, 200))


###
n_c = 504208.0
n_levels = [3e5, n_c, 7e5]
n_levels = [n_c]


print 'T_levels, mu_levels, n_levels: ', T_levels, len(T_levels), mu_levels, len(mu_levels), n_levels, len(n_levels)
invalid_levels = []

#++++++++++++++
calc_const_T_curves  = 0
calc_const_mu_curves = 0
calc_const_n_curves  = 1
save_TD_lvls = 0        # save Thermo along T and mu lines

save_metric = 0         # save metric solutions for HEE calculation
calculate_hee_lvls = 0  # numerical calculation of hee for various mu levels
save_hee_lvls = 0       # save pickles w/ hee on mu levels

save_figs = 0           # saves only figure for crit exp alpha in const_n calculation
#++++++++++++++


cnt1 = contour(Phi1r_raster, lphi0_raster, TD_grid[:, :, 0], levels = T_levels, cmap = cm.autumn)
cnt2 = contour(Phi1r_raster, lphi0_raster, TD_grid[:, :, 2], levels = mu_levels, cmap = cm.winter)
show()
cnt3 = contour(Phi1r_raster, np.exp(lphi0_raster), TD_grid[:, :, 3], levels = n_levels, cmap = cm.winter)
show()

xy_rev = 1
path_rev = 1
rem_lr = 1
cT_contours  = get_contours(cnt1, xy_rev, path_rev, T_levels, rem_lr, 'T')
cmu_contours = get_contours(cnt2, xy_rev, 0, mu_levels, rem_lr, 'mu')
cn_contours  = cnt3.allsegs[0][0] # get_contours(cnt3, xy_rev, 0, n_levels, rem_lr, 'mu')

derv_cutoff = 8.0
#derv_cutoff = 5.0

print 'cT_contours: ', cT_contours
print 'cn_contours: ', cn_contours, len(cn_contours[:,0])
#plot(cn_contours[:,0], cn_contours[:,1])
#show()


num_pts = 90*2
acc_cntrs = {'T':{}, 'mu':{}}

metric_dic = {}
HEE_mulvls = {}
HEE_ren_mulvls = {}


###############################################################################
cnt1_colors = [None]*len(cnt1.collections)
for lind in range(0, len(cnt1.collections)):
    cnt1_colors[lind] = cnt1.collections[lind].get_colors()[0]
cnt2_colors = [None]*len(cnt2.collections)
for lind in range(0, len(cnt2.collections)):
    cnt2_colors[lind] = cnt2.collections[lind].get_colors()[0]

print len(cnt1_colors), len(cnt2_colors)
j = 0
if calc_const_mu_curves:
    print 'calculating mu = const contours'
    if model_type == 'G':
        lphi0_cutoff = 0.4
    else:
        lphi0_cutoff = -10.0 ## effectively no log(phi_0) cutoff

    iterate = 0
    for mu_cont_val in mu_levels:
        print 4*'#', mu_cont_val
        print 'cmu_contours[1][mu_cont_val]: ', cmu_contours[1][mu_cont_val]
        print 'buh'
        if not (model_type == 'VRY_2' or model_type == 'VRY_4'):
            if len(cmu_contours[1][mu_cont_val][0]) != 2:
                print     cmu_contours[1][mu_cont_val][0][:, 0]
                Phi1r_0 = cmu_contours[1][mu_cont_val][0][:, 1]
                lphi0_0 = cmu_contours[1][mu_cont_val][0][:, 0]
                Phi1r_0 = np.compress(lphi0_0 > lphi0_cutoff, Phi1r_0)
                lphi0_0 = np.compress(lphi0_0 > lphi0_cutoff, lphi0_0)
                cnt_0 = np.transpose(np.vstack((lphi0_0, Phi1r_0)))
            else:
                lphi0_l = cmu_contours[1][mu_cont_val][:, 0]
                print lphi0_l
                Phi1r_0l = np.compress(lphi0_l > lphi0_cutoff, cmu_contours[1][mu_cont_val][:, 1])
                lphi0_0l = np.compress(lphi0_l > lphi0_cutoff, cmu_contours[1][mu_cont_val][:, 0])
                cmu_contours[1][mu_cont_val] = np.transpose(np.vstack((lphi0_0l, Phi1r_0l)))
                print cmu_contours[1]
                cnt_0 = cmu_contours[1][mu_cont_val]
        else:
            cnt_0 = cmu_contours[1][mu_cont_val][0]

        print 'cnt_0: ', cnt_0
        print 4*'#'
        if mu_cont_val > 0:
            mucont_new = get_more_mu_contourpoints(num_pts, cnt_0, derv_cutoff)
        elif mu_cont_val == 0:
            #print cmu_contours
            lphi0_appr = cmu_contours[1][mu_cont_val][0][:, 0]
            lphi0_appr = np.compress(lphi0_appr > lphi0_cutoff, lphi0_appr)
            lphi0_new = np.linspace(lphi0_appr[0], lphi0_appr[-1], num_pts)
            Phi1r_new  = np.zeros(num_pts)

            fitwhat = np.zeros((2, num_pts))
            fitwhat[0, :] = np.ones(num_pts)
            mucont_new = [Phi1r_new, lphi0_new, fitwhat]

        print mucont_new[0], mucont_new[1], mucont_new[2]
        print len(mucont_new[0]), len(mucont_new[1])

        acc_cntrTD = get_acc_contour_and_TD(mucont_new, mu_cont_val, 'mu', V_args, f_args)

        if acc_cntrTD == 0: # in case there is no BH solution
            print '\n+++problematic mu_level: ', mu_cont_val, '\n+++'
            invalid_levels.append(mu_cont_val)
            j += 1
            continue # go to next mu_level

        acc_cntrs['mu'].update({mu_cont_val:acc_cntrTD[0:-1]})

        TD_slice_scaled = TD_scale_isen(acc_cntrTD[1], lambdas)[0]
        #print 'TD_slice_scaled: ', TD_slice_scaled


        if save_metric:
            metric_dic_lvl = acc_cntrTD[2]
            #print 'metric_dic_lvl: ', metric_dic_lvl, metric_dic_lvl.keys()
            metric_dic[mu_cont_val] = {}
            for k in range(0, num_pts):
                T = TD_slice_scaled[:,0][k]
                metric_dic[mu_cont_val].update( {T:{'r':metric_dic_lvl[k]['r'], 'A':metric_dic_lvl[k]['A'], 'h':metric_dic_lvl[k]['h']}} ) #  to get: metric_dic[mu][T] = metric_sol

        if calculate_hee_lvls:
            metric_dic_lvl = acc_cntrTD[2]
            T_list = []
            S_HEE_list = []
            S_HEE_ren_list = []
            for k in range(0, num_pts):
                T = TD_slice_scaled[:,0][k]
                iterate = iterate + 1
                print '\nmu, T     = ', mu_cont_val, T, '   ', iterate, '/', len(mu_levels)*num_pts
                r_vals = metric_dic_lvl[k]['r']
                A_vals = metric_dic_lvl[k]['A']
                h_vals = metric_dic_lvl[k]['h']

                try:
                    hee = hee_integration(r_vals, A_vals, h_vals)   # contains hee integral evaluations

                    S_HEE = hee[0]
                    S_HEE_ren = hee[1]
                    print 'S_HEE     = ', S_HEE
                    print 'S_HEE_ren = ', S_HEE_ren
                    T_list.append(T)
                    S_HEE_list.append(S_HEE)
                    S_HEE_ren_list.append(S_HEE_ren)
                except ValueError:
                    print 'discarded'
                    continue

            ## update HEE_mulvls dictionary for current mu level:
            HEE_mulvls[mu_cont_val] = {'T':np.array(T_list), 'S_HEE':np.array(S_HEE_list)}
            HEE_ren_mulvls[mu_cont_val] = {'T':np.array(T_list), 'S_HEE_ren':np.array(S_HEE_ren_list)}

        figure(1)
        plot(TD_slice_scaled[:, 0], TD_slice_scaled[:, 1]/TD_slice_scaled[:, 0]**3.0, label = r'$\mu = '+str(mu_cont_val)+'$ [MeV]', color = cnt2_colors[j])

        figure(2)
        plot(TD_slice_scaled[:, 0], TD_slice_scaled[:, 3]/TD_slice_scaled[:, 0]**3.0, label = r'$\mu = '+str(mu_cont_val)+'$ [MeV]', color = cnt2_colors[j])

        j += 1



figure(1)
xlabel(r'$T$ [MeV]')
ylabel(r'$s/T^3$')
legend(loc = 'lower right')

figure(2)
xlabel(r'$T$ [MeV]')
ylabel(r'$\rho/T^3$')
legend()

chi2_array = np.zeros(len(T_levels))
chi2_mu400_array = np.zeros(len(T_levels))
chi3_array = np.zeros(len(T_levels))
chi4_array = np.zeros(len(T_levels))
j = 0
#for T_cont_val in np.arange(50, 255, 50):
if calc_const_T_curves:
    print 'calculating T = const contours'
    for T_cont_val in T_levels:
        print 'cT_contours[1][T_cont_val]: ', cT_contours[1][T_cont_val]
        print     cT_contours[1][T_cont_val][0][:, 0]
        lphi0_0 = cT_contours[1][T_cont_val][0][:, 0]
        Phi1r_0 = cT_contours[1][T_cont_val][0][:, 1]
        if T_cont_val >= 150:
            print 4*'#'
            Phir_cutoff = 0.74
            lphi0_0 = np.compress(Phi1r_0 < Phir_cutoff, lphi0_0)
            Phi1r_0 = np.compress(Phi1r_0 < Phir_cutoff, Phi1r_0)
            cnt_0 = np.transpose(np.vstack((lphi0_0, Phi1r_0)))
            print 'cnt_0: ', cnt_0
            print 4*'#'
            Tcont_new = get_more_T_contourpoints(num_pts, cnt_0, derv_cutoff)
    #    if T_cont_val == 100 or T_cont_val == 50:
    #        lphi0_0 = np.compress(Phi1r_0 > 0.1, lphi0_0)
    #        Phi1r_0 = np.compress(Phi1r_0 > 0.1, Phi1r_0)
    #        cnt_0 = np.transpose(np.vstack((lphi0_0, Phi1r_0)))
    #        Tcont_new = get_more_T_contourpoints(num_pts, cnt_0, derv_cutoff)
        else:
            Tcont_new = get_more_T_contourpoints(num_pts, cT_contours[1][T_cont_val][0], derv_cutoff)

        print 'Tcont_new: ', Tcont_new[0], Tcont_new[1], Tcont_new[2]
        print len(Tcont_new[0]), len(Tcont_new[1])

        acc_cntrTD = get_acc_contour_and_TD(Tcont_new, T_cont_val, 'T', V_args, f_args)

        if acc_cntrTD == 0: # in case there is no BH solution
            print '\n+++problematic T_level: ', T_cont_val, '\n+++'
            invalid_levels.append(T_cont_val)
            j += 1
            continue # go to next T_level

        acc_cntrs['T'].update({T_cont_val:acc_cntrTD[0:-1]})

        TD_slice_scaled = TD_scale_isen(acc_cntrTD[1], lambdas)[0]

        ## chis:
        mu_help, rho_help = zip(*sorted(zip(TD_slice_scaled[:, 2], TD_slice_scaled[:, 3])))
        chi2_array[j] = splev(np.array([0.0]), splrep(mu_help, rho_help), der=1)
        chi2_mu400_array[j] = splev(np.array([400.0]), splrep(mu_help, rho_help), der=1)
        chi3_array[j] = splev(np.array([0.0]), splrep(mu_help, rho_help), der=2)
        chi4_array[j] = splev(np.array([0.0]), splrep(mu_help, rho_help), der=3)

        ## Plots:
        figure(3)
        plot(TD_slice_scaled[:, 2], TD_slice_scaled[:, 1]/TD_slice_scaled[:, 2]**3.0, label = r'$T = '+str(T_cont_val)+'$ [MeV]', color = cnt1_colors[j])

        figure(4)
        plot(TD_slice_scaled[:, 2], TD_slice_scaled[:, 3]/TD_slice_scaled[:, 2]**3.0, label = r'$T = '+str(T_cont_val)+'$ [MeV]', color = cnt1_colors[j])

        figure(5)
        plot(TD_slice_scaled[:, 2], TD_slice_scaled[:, 3], label = r'$T = '+str(T_cont_val)+'$ [MeV]', color = cnt1_colors[j])

        j += 1



###--- calc n=const contours
###    C_n ~ (T - T_CEP)^-alpah along FOPT curve, mimicked by n_CEP curve
if calc_const_n_curves:
    T_CEP = 111.5
    s_c = 8371569.12395
    TD_slice = np.zeros((len(cn_contours[:,1]), 4))
    fname = model_type+'/'+ftype+'/Tc_muT.p'
    Tc_muT = pickle.load(open(fname, "rb")) # FOPT curve from 'true' TD
    file.close(open(fname))

    for i in range(0, len(cn_contours[:,1])):
        phi0 = cn_contours[:,1][i]
        Phi1r = cn_contours[:,0][i]
        r_mid = 12.0
        TD_pt = TD_calc_pointwise(phi0, Phi1r, V_args, f_args, r_mid)
        TD_slice[i,:] = TD_pt[0]

    TD_slice_scaled = TD_scale_isen(TD_slice, lambdas)[0]
    TD_slice_scaled_copy = 1*TD_slice_scaled    # full saved version for further calculations
    print 'TD_slice_scaled: ', TD_slice_scaled
    print 'n rel diff, max: ', (TD_slice_scaled[:,3]-n_c)/n_c, np.max(np.abs(TD_slice_scaled[:,3]-n_c)/n_c)
    print 's_c: ', s_c, splev(T_CEP, splrep(TD_slice_scaled[:,0], TD_slice_scaled[:,1])), (s_c - splev(T_CEP, splrep(TD_slice_scaled[:,0], TD_slice_scaled[:,1])))/s_c

    figure(100)
    plot(TD_slice_scaled[:,0], TD_slice_scaled[:,1]) # s(T)
    t = (TD_slice_scaled[:,0]-T_CEP)/T_CEP
    plot(TD_slice_scaled[:,0], TD_slice_scaled[:,0]*splev(TD_slice_scaled[:,0], splrep(TD_slice_scaled[:,0], TD_slice_scaled[:,1]), der=1)) # C_n(T)

    figure(101) # T(mu)
    plot(TD_slice_scaled[:,2], TD_slice_scaled[:,0]) # const n_c
    plot(Tc_muT[0,:], Tc_muT[1,:], ls='--') # FOPT curve

    figure(102) # ln(s-s_c) - ln(T-Tc)
    #plot(np.log(np.abs(TD_slice_scaled[:,1]-s_c)), np.log(np.abs(TD_slice_scaled[:,0]-T_CEP)), ls='', marker='s')
    for i in range(0, len(TD_slice_scaled[:,0])):
        if TD_slice_scaled[i,0] < T_CEP:
            plot(np.log(np.abs(TD_slice_scaled[i,1]-s_c)), np.log(np.abs(TD_slice_scaled[i,0]-T_CEP)), ls='', marker='s', c='k')
        elif TD_slice_scaled[i,0] >= 130.0:
            plot(np.log(np.abs(TD_slice_scaled[i,1]-s_c)), np.log(np.abs(TD_slice_scaled[i,0]-T_CEP)), ls='', marker='s', c='g')
        else:
            plot(np.log(np.abs(TD_slice_scaled[i,1]-s_c)), np.log(np.abs(TD_slice_scaled[i,0]-T_CEP)), ls='', marker='s', c='b')

    ## masking:
    TD_slice_scaled = ma.asarray(TD_slice_scaled)
    for i in range(0, len(TD_slice_scaled[:,0])):
        if TD_slice_scaled[i,0] < T_CEP or TD_slice_scaled[i,0] >= 130.0:
             TD_slice_scaled[i,:] = ma.masked

    ## linear fit:
    ln_s = np.log(np.abs(TD_slice_scaled[:,1]-s_c))
    ln_s = ln_s[~ln_s.mask]
    ln_T = np.log(np.abs(TD_slice_scaled[:,0]-T_CEP))
    ln_T = ln_T[~ln_T.mask]
    linfit_alpha = np.polyfit(ln_s, ln_T, 1)
    linfct_alpha = np.poly1d(linfit_alpha)
    alpha = -(1.0/linfit_alpha[0]-1)
    print '\ncritical exponent linear fit, alpha: ', linfit_alpha, alpha

    figure(103) # ln(s-s_c) - ln(T-Tc)
    plot(ln_s, ln_T, ls='', marker='s')
    plot(ln_s, linfct_alpha(ln_s))
    axis([11, 15, -0.5, 3])
    text(13.7, 1.3, r'$\alpha \approx %1.4f$' %alpha)
    xlabel(r'$\ln\vert s-s_{CEP} \vert$')
    ylabel(r'$\ln\vert T-T_{CEP} \vert$')
    nice_ticks()

    if save_figs:
        savefig(model_type+'/'+ftype+'/pdfs/critexp/'+model_type+'_critexp_alpha.pdf')

    ## saving n_CEP contour line:
    fname = model_type+'/'+ftype+'/nCEP_contour_'+model_type+'.p'
    pickle.dump(TD_slice_scaled_copy, open(fname, "wb"))
    file.close(open(fname))


figure(3)
xlabel(r'$\mu$ [MeV]')
ylabel(r'$s/\mu^3$')
legend(loc = 'upper right')

figure(4)
xlabel(r'$\mu$ [MeV]')
ylabel(r'$\rho/\mu^3$')
legend()

figure(5)
xlabel(r'$\mu$ [MeV]')
ylabel(r'$\rho$ [MeV^3]')
legend()

figure(6)
plot(np.array(T_levels), chi2_array/np.array(T_levels)**2, label=r'$\mu=0$ MeV')
plot(np.array(T_levels), chi2_mu400_array/np.array(T_levels)**2, label=r'$\mu=400$ MeV')
xlabel(r'$T$ [MeV]')
ylabel(r'$\chi_2 / T^2$ ')
legend()

figure(7)
plot(np.array(T_levels), chi3_array/np.array(T_levels))
xlabel(r'$T$ [MeV]')
ylabel(r'$\chi_3 / T$ ')
legend()

figure(8)
plot(np.array(T_levels), chi4_array)
xlabel(r'$T$ [MeV]')
ylabel(r'$\chi_4$ ')
legend()

for i in range(0, 9):
    figure(i)
    ax = subplot(111)
    for l in ax.get_xticklines() + ax.get_yticklines():
        l.set_markersize(8)
        l.set_markeredgewidth(3)



## SAVING:
#chis_dic = {'T_levels':np.array(T_levels), 'chi2T2':chi2_array/np.array(T_levels)**2, 'chi2T2_mu400':chi2_mu400_array/np.array(T_levels)**2, 'chi3T':chi3_array/np.array(T_levels), 'chi4':chi4_array}
#fname = model_type+'/'+ftype+'/chis_alongT_'+model_type+'.p'
#pickle.dump(chis_dic, open(fname, "wb"))
#file.close(open(fname))

if save_TD_lvls:
    #fname = model_type+'/'+ftype+'/T_mu_contours_'+model_type+'_moreTpts.p'
    #fname = model_type+'/'+ftype+'/hee/T_mu_contours_'+model_type+'_mupts_part5.p'
    fname = model_type+'/'+ftype+'/T_mu_contours_'+model_type+'.p'
    fname = model_type+'/'+ftype+'/T_mu_contours_'+model_type+'_muCEP.p'
    fname = model_type+'/'+ftype+'/T_mu_contours_'+model_type+'_TCEP.p'
    pickle.dump(acc_cntrs, open(fname, "wb"))
    file.close(open(fname))

if save_metric:
    fname = model_type+'/'+ftype+'/hee/metric_dic_mulvls.p'
    pickle.dump(metric_dic, open(fname, "wb"))
    file.close(open(fname))

if save_hee_lvls:
    print '\nHEE_mulvls: \n', np.sort(HEE_mulvls.keys())
    fname = model_type+'/'+ftype+'/hee/HEE_mulvls.p'
    pickle.dump(HEE_mulvls, open(fname, "wb"))
    file.close(open(fname))

    fname = model_type+'/'+ftype+'/hee/HEE_ren_mulvls.p'
    pickle.dump(HEE_ren_mulvls, open(fname, "wb"))
    file.close(open(fname))


show()

print 'invalid_levels: ', invalid_levels
print '\ndone: fmg_contours_wTD.py'