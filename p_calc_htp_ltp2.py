import numpy as np
import numpy.ma as ma
from scipy.interpolate import splrep, splev
import pickle
from fmg_p_calc_grid_module3 import p_calc_grid
from args_and_lambds import args_dic

model_type = 'VRY_2'
ftype = args_dic['ftype'][model_type]

fname = model_type+'/'+ftype+'/TD_grid_ltp_htp_VRY_2_2.p'
TD_grid_ltp_htp = pickle.load(open(fname, "rb"))
file.close(open(fname))

fname = model_type+'/'+ftype+'/phiPhi_grid_ltp_htp_VRY_2_2.p'
phiPhi_grid_ltp_htp = pickle.load(open(fname, "rb"))
file.close(open(fname))

fname = model_type+'/'+ftype+'/pc_lines_VRY_2_2.p'
pclines = pickle.load(open(fname, "rb"))
file.close(open(fname))

print pclines['ltp'][1, :]

p_grids_ltp_htp = {}

for ph in ['ltp', 'htp']:
    print ph
    print 20*'#'
    Phip_init = [pclines[ph][1, :], pclines[ph][2, :]]
    print Phip_init
    print Phip_init[1]
    p_grid = p_calc_grid(TD_grid_ltp_htp[ph], phiPhi_grid_ltp_htp[ph], Phip_init, [0])
    print 20*'#'
    print p_grid
    p_grids_ltp_htp.update({ph:p_grid})

fname = model_type+'/'+ftype+'/p_grid_ltp_htp_VRY_2_2.p'
pickle.dump(p_grids_ltp_htp, open(fname, "wb"))
file.close(open(fname))


print '\ndone: p_calc_htp_ltp2.py'