import numpy as np
from fmg_TDprocess import TD_scale, TD_scale_isen, J_calc_fd
from scipy.interpolate import splrep, splev
from scipy.optimize import fmin_l_bfgs_b, brentq
from amoba import amoeba

from pylab import figure, plot, legend, show, semilogy, scatter, xlabel, ylabel, rc, axis, savefig, subplot, contour, contourf, colorbar, cm, axhline, axvline, getp#, add_axes

import pickle
from matplotlib.transforms import Bbox

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

linew = 3
rc('font', size = 20) #fontsize of axis labels (numbers)
rc('axes', labelsize = 20, lw = linew) #fontsize of axis labels (symbols)
rc('lines', mew = 2, lw = linew)
rc('patch', ec = 'k')

rc('legend', fontsize = 16, frameon = True, fancybox = True, columnspacing = 1)
#####################
lambdas = [252.0, 121.0**3.0, 972.0, 77.0**3.0] #T, s, mu, rho

fname = 'TD_gr_G.p'
TD_gr = pickle.load(open(fname, "rb"))

TD_grid = TD_gr[0]
TD_grid = TD_scale(TD_grid, lambdas)[0]
phiPhi_grid = TD_gr[1]
TD_full = TD_gr[2]

phi0_pts = TD_full['phi0_pts']
Phi1_pts = TD_full['Phi1_pts']

lphi0_raster = np.log(TD_full['phi0_raster'])
Phi1r_raster = phiPhi_grid[20, :][:, 1]

print lphi0_raster
print Phi1r_raster
print TD_grid[:, :, 0]

T_levels = list(np.arange(50, 305, 50))
#mu_levels = list(np.arange(250, 2050, 250))
mu_levels = list(np.arange(200, 1650, 200))
print T_levels

#from fmg_plotter_Tmu_wisen5 import fig19, fig20

ls_mu = 'solid'
fig = figure(19)
cnt1 = contour(Phi1r_raster, lphi0_raster, TD_grid[:, :, 0], levels = T_levels, cmap = cm.autumn, zorder = 2)
cnt2 = contour(Phi1r_raster, lphi0_raster, TD_grid[:, :, 2], levels = mu_levels, cmap = cm.winter, linestyles = ls_mu, zorder = 1)

ax = subplot(111)
ax.set_position(Bbox([[0.12, 0.02], [0.78, 0.9]]))
cb = colorbar(cnt2, ax = ax, orientation = 'horizontal', pad = 0.135)
cb.ax.set_ylabel(r'$\mu$', rotation = 0)
#cb.ax.set_xlim(0, 1.0)

ax2 = fig.add_axes([0.82, 0.108, 0.035, 0.795])
cb2 = fig.colorbar(cnt1, cax=ax2)
cb2.ax.set_ylabel(r'$T$', rotation = 90)

cb2.ax.get_children()[4].set_linestyle(ls_mu)

cb.ax.get_children()[4].set_linewidths(8)
cb2.ax.get_children()[4].set_linewidths(8)

cnt1_colors = [None]*len(cnt1.collections)
for lind in range(0, len(cnt1.collections)):
    cnt1_colors[lind] = cnt1.collections[lind].get_colors()[0]
cnt2_colors = [None]*len(cnt2.collections)
for lind in range(0, len(cnt2.collections)):
    cnt2_colors[lind] = cnt2.collections[lind].get_colors()[0]

figure(20)
jT = 0
jmu = 0
for lvl in T_levels:
    axhline(lvl, color = cnt1_colors[jT], zorder = -15)
    jT += 1
for lvl in mu_levels:
    axvline(lvl, color = cnt2_colors[jmu], zorder = -20, ls = ls_mu)
    jmu += 1

#axis([0, 1950, 0, 250.001])

def get_contours(cnt, xy_rev, path_rev, levels, rem_lr):
    cont_list = [None]*len(levels)
    cont_dic = {}
    gcv = get_contour_verts(cnt, xy_rev, path_rev)

    for k in range(0, len(levels)):
        seg_num = len(gcv[k])
        if rem_lr:
            if seg_num == 2:
                cont_dic.update({levels[k]:gcv[k][1]})
                cont_list[k] = gcv[k][1]
            elif seg_num == 3:
                cont_dic.update({levels[k]:[gcv[k][0], gcv[k][2]]})
                cont_list[k] = [gcv[k][0], gcv[k][2]]
        else:
            cont_dic.update({levels[k]:gcv[k]})
            cont_list[k] = gcv[k]
        print levels[k], 'index length of contour', len(cont_dic[levels[k]])

    return [cont_list, cont_dic]

print 'extracted contours:'
xy_rev = 1
path_rev = 1
rem_lr = 1
cT_contours = get_contours(cnt1, xy_rev, path_rev, T_levels, rem_lr)
#print cT_contours[0][0]
#print cT_contours[0][0][:, 0]

#print cmu_contours[1][1600]
#for key in cmu_contours[1].keys():
#    print key, cmu_contours[1][key]

#from backbone_g_4dl2 import TD_calc_pointwise
from TD_calc_pow import TD_calc_pointwise

def get_more_T_contourpoints(num_pts, cntr_approx, derv_cutoff):
    lphi0_appr = cntr_approx[:, 0]
    Phi1r_appr = cntr_approx[:, 1]
    npo = len(lphi0_appr)
    print npo
    derv_raster = np.zeros(npo)
    fitwhat = np.zeros((2, num_pts))
    ##calculate finite difference derivative and check if T contour exhibits reverse-s shape or not
    for i in range(0, npo - 1):
        i_sw1 = npo - 1
        derv_raster[i] = (lphi0_appr[i + 1] - lphi0_appr[i])/(Phi1r_appr[i + 1] - Phi1r_appr[i])
    derv_raster[-1] = (lphi0_appr[-1] - lphi0_appr[-2])/(Phi1r_appr[-1] - Phi1r_appr[-2])

    print 'min/max deriv:', np.amin(derv_raster), np.amax(np.abs(derv_raster))
    if np.amax(np.abs(derv_raster)) < 100: ##no reverse s-shape
        Phi1r_new = np.linspace(Phi1r_appr[0], Phi1r_appr[-1], num_pts)
        cntr_tck = splrep(Phi1r_appr, lphi0_appr, k = 1) ##no rss => no probs with spline interpolation
        lphi0_new = splev(Phi1r_new, cntr_tck)
        fitwhat[1, :] = np.ones(num_pts) ##keep Phi1r fixed and fit lphi0

    else: ##reverse s-shape
        for i in range(0, npo):
            if derv_raster[i] < - derv_cutoff:
                i_sw1 = i
                break
        for i in range(i_sw1, npo):
            if derv_raster[i] < 0 and derv_raster[i] > - derv_cutoff:
                i_sw2 = i
                break
        print i_sw1, i_sw2
        print lphi0_appr[i_sw1], lphi0_appr[i_sw2]

        Phi1r_new1 = np.linspace(Phi1r_appr[0], Phi1r_appr[i_sw1 - 1], int(num_pts/2.0))
        cntr_tck1 = splrep(Phi1r_appr[:i_sw1], lphi0_appr[:i_sw1]) ##no rss => no probs with spline interpolation
        lphi0_new1 = splev(Phi1r_new1, cntr_tck1)
        fitwhat[1, :int(num_pts/2.0)] = np.ones(int(num_pts/2.0))

        lphi0_new2 = np.linspace(lphi0_appr[i_sw1], lphi0_appr[i_sw2 - 1], int(num_pts/3.0))
        cntr_tck2 = splrep(lphi0_appr[i_sw1:i_sw2][::-1], Phi1r_appr[i_sw1:i_sw2][::-1])
        Phi1r_new2 = splev(lphi0_new2, cntr_tck2)
        fitwhat[0, int(num_pts/2.0):5.0*int(num_pts/6.0)] = np.ones(int(num_pts/3.0))

        Phi1r_new3 = np.linspace(Phi1r_appr[i_sw2], Phi1r_appr[-1], int(num_pts/6.0))
        cntr_tck3 = splrep(Phi1r_appr[i_sw2:], np.log(lphi0_appr[i_sw2:]), k =1) ##no rss => no probs with spline interpolation
        lphi0_new3 = np.exp(splev(Phi1r_new3, cntr_tck3))
        fitwhat[1, 5.0*int(num_pts/6.0):] = np.ones(int(num_pts/6.0))

        Phi1r_new = np.hstack((Phi1r_new1, Phi1r_new2, Phi1r_new3))
        lphi0_new = np.hstack((lphi0_new1, lphi0_new2, lphi0_new3))

    return [Phi1r_new, lphi0_new, fitwhat]

#Tcont_new = get_more_T_contourpoints(120, cT_contours[1][100], derv_cutoff)
#print Tcont_new

#figure(21)
#plot(Tcont_new[0], Tcont_new[1])

def get_more_mu_contourpoints(num_pts, cntr_approx, derv_cutoff):
    if len(cntr_approx) != 2:
        lphi0_appr = cntr_approx[:, 0]
        Phi1r_appr = cntr_approx[:, 1]
        print lphi0_appr, Phi1r_appr
        fitwhat = np.zeros((2, num_pts))

        if np.amax(Phi1r_appr) < 0.7: ##simplest case -> take lphi0 as good monotonous coord
            lphi0_new = np.linspace(lphi0_appr[0], lphi0_appr[- 1], num_pts)
            cntr_tck = splrep(lphi0_appr, Phi1r_appr, k = 3)
            Phi1r_new = splev(lphi0_new, cntr_tck)
            fitwhat[0, :] = np.ones(num_pts)

        elif np.amax(Phi1r_appr) >= 0.7:
            print 'buh'
            Phi1r_argmax = np.argmax(Phi1r_appr)

            lphi0_appr_l = lphi0_appr[:Phi1r_argmax]
            Phi1r_appr_l = Phi1r_appr[:Phi1r_argmax]
            fitwhat_l = np.zeros((2, num_pts/2.0))

            lphi0_appr_u = lphi0_appr[Phi1r_argmax:]
            Phi1r_appr_u = Phi1r_appr[Phi1r_argmax:]
            fitwhat_u = np.zeros((2, num_pts/2.0))

            Phi1r_new_l = np.linspace(Phi1r_appr_l[0], Phi1r_appr_l[-1], int(num_pts/2.0))
            cntr_l_tck = splrep(Phi1r_appr_l, lphi0_appr_l, k = 3)
            lphi0_new_l = splev(Phi1r_new_l, cntr_l_tck)
            fitwhat_l[1, :] = np.ones(int(num_pts/2.0))

            Phi1r_new_u = np.linspace(Phi1r_appr_u[0], Phi1r_appr_u[-1], int(num_pts/2.0))
            cntr_u_tck = splrep(Phi1r_appr_u[::-1], lphi0_appr_u[::-1], k = 3)
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


def phi_or_Phi_fit(fit_fld, fixfld, fitwhat, TD_val, whichfunc_ind, V_args, f_args, r_mid):
    fit_fld = fit_fld[0]
    phi0 = np.array([fit_fld, fixfld])[fitwhat.argmin()]
    Phi1 = np.dot(np.array([fit_fld, fixfld]), fitwhat)
    #print fixfld, fit_fld, phi0, Phi1, whichfunc_ind

    TD_pt = TD_calc_pointwise(phi0, Phi1, V_args, f_args, r_mid)
    func_val = TD_pt[0][whichfunc_ind]*lambdas[whichfunc_ind]
    print 4*'#'
    print 'phi_0 = %2.12f, Phi_1 = %2.12f, func_val = %2.8f, TD_val = %2.8f' %(phi0, Phi1, func_val, TD_val) #'TD_pt[0][whichfunc_ind]

    return (func_val - TD_val)**2.0

def phi_or_Phi_fit_a(fit_fld, data):
    fixfld = data[0]
    fitwhat = data[1]
    TD_val = data[2]
    whichfunc_ind = data[3]
    V_args = data[4]
    f_args = data[5]
    r_mid = data[6]

    fit_fld = fit_fld[0]

    #print fixfld, fit_fld
    phi0 = np.array([fit_fld, fixfld])[fitwhat.argmin()]
    Phi1 = np.dot(np.array([fit_fld, fixfld]), fitwhat)
    print fixfld, fit_fld, phi0, Phi1, whichfunc_ind

    TD_pt = TD_calc_pointwise(phi0, Phi1, V_args, f_args, r_mid)
    func_val = TD_pt[0][whichfunc_ind]*lambdas[whichfunc_ind]
    print 4*'#'
    print 'phi_0 = %2.12f, Phi_1r = %2.12f, func_val = %2.8f, TD_val = %2.8f' %(phi0, Phi1, func_val, TD_val) #'TD_pt[0][whichfunc_ind]
    print 4*'#'
    chiq = - np.log((func_val - TD_val)**2.0)
    print chiq

    return chiq

def phi_or_Phi_fit_b(fit_fld, fixfld, fitwhat, TD_val, whichfunc_ind, V_args, f_args, r_mid):
    #fit_fld = fit_fld[0]

    #print fixfld, fit_fld
    phi0 = np.array([fit_fld, fixfld])[fitwhat.argmin()]
    Phi1 = np.dot(np.array([fit_fld, fixfld]), fitwhat)
    print fixfld, fit_fld, phi0, Phi1, whichfunc_ind

    TD_pt = TD_calc_pointwise(phi0, Phi1, V_args, f_args, r_mid)
    func_val = TD_pt[0][whichfunc_ind]*lambdas[whichfunc_ind]
    print 4*'#'
    print 'phi_0 = %2.12f, Phi_1r = %2.12f, func_val = %2.8f, TD_val = %2.8f' %(phi0, Phi1, func_val, TD_val) #'TD_pt[0][whichfunc_ind]
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

    print Phiphi_acc[:, 0]
    #print fitwhat[:, 0]
    TD_inds = {'T':0, 's':1, 'mu':2, 'rho':3}
    fields_raster = np.vstack((Phi1r_raster, phi0_raster))
    r_mid = 12.0

    for i in range(0, len(Phi1r_raster)):
        fields = np.array([Phi1r_raster[i], phi0_raster[i]])
        fix_field = fields[fitwhat[:, i].argmin()]
        fit_field_init = np.dot(fields, fitwhat[:, i])
        print fields, i, fitwhat[:, i], fix_field, fit_field_init

        Phiphi_acc[fitwhat[:, i].argmin(), i] = fix_field
        print Phiphi_acc[:, i]

        ##bounds:
        if i == 0 or i == len(Phi1r_raster) - 1:
            bnds = [(fit_field_init*0.99, fit_field_init*1.01)]
            #print bnds
        else:
            #print fields_raster[:, i - 1]
            b1 = np.dot(fields_raster[:, i - 1], fitwhat[:, i])
            b2 = np.dot(fields_raster[:, i + 1], fitwhat[:, i])
            bnds = [(np.sort([b1, b2]))]
        print 'bnds =', bnds

        #fitted_field = fmin_l_bfgs_b(phi_or_Phi_fit, fit_field_init, bounds = bnds, args = (fix_field, fitwhat[:, i], cnt_val, TD_inds[TDfunc], V_args, f_args, 12.0), approx_grad = True)[0]
        #fitted_field = amoeba([fit_field_init], [fit_field_init/200.0], phi_or_Phi_fit_a, xtolerance=1e-3, ftolerance = 5.0*1e-3, data = [fix_field, fitwhat[:, i], cnt_val, TD_inds[TDfunc], V_args, f_args, 12.0])[0]
        a_fac = 0.99
        b_fac = 1.01
        if i == 0:
            a_fac = 0.95
            b_fac = 1.05
        fitted_field = brentq(phi_or_Phi_fit_b, fit_field_init*a_fac, fit_field_init*b_fac, xtol=1e-9, rtol=1e-8, args = (fix_field, fitwhat[:, i], cnt_val, TD_inds[TDfunc], V_args, f_args, r_mid))#[0]
        print 10*'#'
        print fitted_field
        print 10*'#'

        phi0 = np.array([fitted_field, fix_field])[fitwhat[:, i].argmin()]
        Phi1r = np.dot(np.array([fitted_field, fix_field]), fitwhat[:, i])
        TD_pt = TD_calc_pointwise(phi0, Phi1r, V_args, f_args, r_mid)

        TD_slice[i, :] = TD_pt[0]
        r_mid = TD_pt[2]

        Phiphi_acc[fitwhat[:, i].argmax(), i] = fitted_field#[0]

    return [Phiphi_acc, TD_slice]

gamma = 0.606
b = 2.057
V_args = ['V_I', np.array([gamma, b])]

##f_I = nrm*1/cosh(scl*(phi - shft))
nrm = np.cosh(12.0/5.0)
scl = 6.0/5.0
shft = 2.0
f_args = ['f_I', np.array([nrm, scl, shft])]

#############
#mu_levels = list(np.arange(400, 2050, 200))
derv_cutoff = 8.0
cnt_val = 800

cmu_contours = get_contours(cnt2, xy_rev, 0, mu_levels, rem_lr)

for cnt_val in np.arange(200, 1050, 200):
    mucont_new = get_more_mu_contourpoints(60, cmu_contours[1][cnt_val], derv_cutoff)

    print mucont_new[0], mucont_new[1], mucont_new[2]
    print len(mucont_new[0]), len(mucont_new[1])

    acc_cntrTD = get_acc_contour_and_TD(mucont_new, cnt_val, 'mu', V_args, f_args)

    figure(19)
    plot(acc_cntrTD[0][0, :], np.log(acc_cntrTD[0][1, :]), ls = 'dashed')

    TD_slice_scaled = TD_scale_isen(acc_cntrTD[1], lambdas)[0]

    figure(1)
    plot(TD_slice_scaled[:, 0], TD_slice_scaled[:, 1]/TD_slice_scaled[:, 0]**3.0, label = r'$\mu = '+str(cnt_val)+'$')

    figure(2)
    plot(TD_slice_scaled[:, 0], TD_slice_scaled[:, 3]/TD_slice_scaled[:, 0]**3.0, label = r'$\mu = '+str(cnt_val)+'$')


figure(1)
xlabel(r'$T$ [MeV]')
ylabel(r'$s/T^3$')
legend(loc = 'lower right')

figure(2)
xlabel(r'$T$ [MeV]')
ylabel(r'$\rho/T^3$')
legend()

show()

