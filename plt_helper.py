import numpy as np
from fmg_TDprocess import TD_scale, TD_scale_isen, J_calc_fd
from scipy.interpolate import splrep, splev
from scipy.optimize import fmin_l_bfgs_b, brentq
from amoba import amoeba

from pylab import figure, plot, legend, show, semilogy, scatter, xlabel, ylabel, rc, axis, savefig, subplot, contour, contourf, colorbar, cm, axhline, axvline, getp#, add_axes

from TD_calc_pow import TD_calc_pointwise

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
        #print seg_num, levels[k]
        if rem_lr:
            if seg_num == 2:
                #if cnt > 0:
                #print gcv[k]
                cont_dic.update({levels[k]:gcv[k][exind]})
                cont_list[k] = gcv[k][exind]
            elif seg_num == 3:
                cont_dic.update({levels[k]:[gcv[k][0], gcv[k][2]]})
                cont_list[k] = [gcv[k][0], gcv[k][2]]
        else:
            cont_dic.update({levels[k]:gcv[k]})
            cont_list[k] = gcv[k]
        print levels[k], 'index length of contour', len(cont_dic[levels[k]])

    return [cont_list, cont_dic]
