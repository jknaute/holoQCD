import numpy as np
from Vtypes import v_13, dv_13, v_rpdb_pl, dv_rpdb_pl, V_13, V_rpdb_pl
from ftypes import f_tanh
from scipy.optimize import fsolve

def F(x, *args):
    phi_m = args[0]
    dlogV_l = args[1]
    ddlogV_l = args[2]
    dlogV_r = args[3]
    ddlogV_r = args[4]
    return [dlogV_l(phi_m, x[0], x[1]) - dlogV_r, ddlogV_l(phi_m, x[0], x[1]) - ddlogV_r]

def p_13_from_phim(pars):
    phi_m = pars[0]
    V_pars = pars[1]
    dlogV_m1 = v_rpdb_pl(phi_m, *V_pars)
    ddlogV_m1 = dv_rpdb_pl(phi_m, *V_pars)

    s_13 = fsolve(F, [0.3, 0.5], args = (phi_m, v_13, dv_13, dlogV_m1, ddlogV_m1), xtol = 1e-12)
    mq = -12.0*s_13[0]
    Delta = 2.0 + np.sqrt(4.0 + mq)
    print s_13, mq, Delta

    return s_13




################  V and f args  ###############
### Gubser: ###
gamma = 0.606 # 0.606
b = 2.057 # 2.057
V_args_G = ['V_I', np.array([gamma, b])]

##f_I = nrm*1/cosh(scl*(phi - shft))
nrm = np.cosh(12.0/5.0)
scl = 6.0/5.0
shft = 2.0
f_args_G = ['f_I', np.array([nrm, scl, shft])]





### Noronha: ###
gamma = 0.606 # 0.59480153 # 0.606
b     = 0.703 # 0.7958487 # 0.703
c_4   = -0.1 # -0.10434335 # -0.1
c_6   = 0.0034 # 0.00443893 # 0.0034
V_args_no = ['V_VI', np.array([gamma, b, c_4, c_6])]

##f_I = nrm*1/cosh(scl*(phi - shft)) + n2*e^(efac*phi)
nrm  = 1.0/3.0*np.cosh(0.69) # 0.38662661
scl  = 1.2 # 0.8348755
shft = 0.69/1.2 # 0.70053742
n2   = 2.0/3.0 # 0.67281478
efac = -100.0 # -73.39025236
f_args_no = ['f_no', np.array([nrm, scl, shft, n2, efac])]
#f_args_no = ['f_tanh', np.array([116.8271068 , -116.05989068, 1.20542011, 2.31387902])] # for f_tanh type





### Finazzo: ###
V_args_fi = ['V_VI', np.array([0.63, 0.65, -0.05, 0.003])]
f_args_fi = ['f_fi', np.array([0.95, 0.22, -0.15, -0.32])] # [nrm, c2, c1, c0]
#f_args_fi = ['f_fi', np.array([0.97141876,  0.50031315, -0.69095488,  0.4867331])] # fitted

Lambda_fi = 1058.83
kappa_5_fi = 8.0*np.pi*0.46
lambdas_fi = [Lambda_fi, 1.0/kappa_5_fi*Lambda_fi**3.0, Lambda_fi, 1.0/kappa_5_fi*Lambda_fi**3.0]
#lambdas_fi = [1059.49951172, 470.012077411**3, 1745.36178819, 387.774312971**3] # fitted




### NOnew - new fit and adjustment of Noronha-type: ###
V_args_NOnew = V_args_fi
f_args_NOnew = ['f_sech', np.array([-0.27, 0.4, 1.7, 100.0])]
lambdas_NOnew = lambdas_fi




###  RY_2:  ###
V_args_proto = [1.7058, np.array([0.7065, 0.4951, 0.0872/0.4951, -0.0113, -0.4701, 2.1420, 4.3150])]
p_13 = p_13_from_phim(V_args_proto)
V_rpdb_pl_phim = V_rpdb_pl(V_args_proto[0], *np.hstack((V_args_proto[1], 1.0)))
V_phim = V_13(V_args_proto[0], *p_13)/V_rpdb_pl_phim
V_args_RY_2 = ['V_rpdb_pl_s', np.hstack((V_args_proto[0], p_13, V_args_proto[1], V_phim))]
print "V_args_RY_2: ", V_args_RY_2



### VRY_3 ###
V_args_RY_3 = V_args_RY_2
f_args_RY_3 = ['f_fi', np.array([1.20598548,  0.48723703, -0.68963106,  0.49461131])]
lambdas_RY_3 = [1148.06518555, 513.010128995**3, 1849.42033208, 417.87217804**3]


### VRY_4 ###
V_args_RY_4 = V_args_RY_2
lambdas_RY_4 = [1148.06518555, 513.010128995**3, 1148.06518555, 513.010128995**3]
#print 'V_args_RY_4: ', V_args_RY_4
#print 'lambdas_RY_4: ', lambdas_RY_4




## f = f_tanh = args[0] + args[1]*np.tanh(args[2]*(t - args[3]))
##--------------
ftype = 'ftype1'
##--------------
if  ftype == 'ftype1':      # 'normal' type
    f_args_RY_2 = ['f_tanh', np.array([2.93377*0.06449, -2.57227*0.06449, 1.54966, 2.18202])]
    f_args_RY_4 = ['f_tanhexp', np.array([2.93377*0.06449, -2.57227*0.06449, 1.54966, 2.18202, 0.621948242188, 112.713623047])]
elif  ftype == 'ftype2':    # slowly going to zero
    f_args_RY_2 = ['f_tanh2', np.array([2.93377*0.06449, -2.57227*0.06449, 1.54966, 2.18202, -0.00515, 4.62, 0.34, 0.0, 0.0])]
    f_args_RY_4 = ['f_tanh2', np.array([2.93377*0.06449, -2.57227*0.06449, 1.54966, 2.18202, -0.00515, 4.62, 0.34, 0.621948242188, 112.713623047])]
elif  ftype == 'ftype3':    # rapidly going to zero
    f_args_RY_2 = ['f_tanh3', np.array([2.93377*0.06449, -2.57227*0.06449, 1.54966, 2.18202, 8.0, 3.34, 0.0, 0.0])]
    f_args_RY_4 = ['f_tanh3', np.array([2.93377*0.06449, -2.57227*0.06449, 1.54966, 2.18202, 1.48, 3.61, 0.621948242188, 112.713623047])] # rapidly going to zero
elif  ftype == 'ftype4': # max V, f and lambda param. variations
    f_args_RY_2 = [] # dummy
    # VRY_4:
    fac = 1.0
    f_args_RY_4 = ['f_tanhexp', fac*np.array([2.93377*0.06449, -2.57227*0.06449, 1.54966, 2.18202, 0.621948242188, 112.713623047])]
    fac = 0.992
    V_args_proto = [fac*1.7058, fac*np.array([0.7065, 0.4951, 0.0872/0.4951, -0.0113, -0.4701, 2.1420, 4.3150])]
    p_13 = p_13_from_phim(V_args_proto)
    V_rpdb_pl_phim = V_rpdb_pl(V_args_proto[0], *np.hstack((V_args_proto[1], 1.0)))
    V_phim = V_13(V_args_proto[0], *p_13)/V_rpdb_pl_phim
    V_args_RY_4 = ['V_rpdb_pl_s', np.hstack((V_args_proto[0], p_13, V_args_proto[1], V_phim))]
    fac = 1.005
    lambdas_RY_4 = [fac*1148.06518555, 513.010128995**3, fac*1148.06518555, 513.010128995**3]
elif  ftype == 'ftype5': # actually also Vtype changed at low T
    f_args_RY_2 = [] # dummy
    fac = 1.0
    f_args_RY_4 = ['f_tanhexp', fac*np.array([2.93377*0.06449, -2.57227*0.06449, 1.54966, 2.18202, 0.621948242188, 112.713623047])]
    f_args_RY_4 = ['f_tanh3', np.array([2.93377*0.06449, -2.57227*0.06449, 1.54966, 2.18202, 1.48, 3.61, 0.621948242188, 112.713623047])] # rapidly going to zero
    # Vtype2:
    fac = 1.0
    a9 = 4.3150
    V_args_proto = [fac*1.7058, fac*np.array([0.7065, 0.4951, 0.0872/0.4951, -0.0113, -0.4701, 2.1420, a9])]
    p_13 = p_13_from_phim(V_args_proto)
    V_rpdb_pl_phim = V_rpdb_pl(V_args_proto[0], *np.hstack((V_args_proto[1], 1.0)))
    V_phim = V_13(V_args_proto[0], *p_13)/V_rpdb_pl_phim
    V_args_RY_4 = ['V_lin', np.hstack((V_args_proto[0], p_13, V_args_proto[1], V_phim))]
    ### exp term: (comment out if not used)
    #param_exp = [3.795, 1.95, 3.02]
    #V_args_RY_4 = ['V_exp', np.hstack((V_args_proto[0], p_13, V_args_proto[1], V_phim, param_exp))]
#print 'f_args_RY_4: ', f_args_RY_4




args_dic = {'V':{'G':V_args_G, 'no':V_args_no, 'fi':V_args_fi, 'VRY_2':V_args_RY_2, 'NOnew':V_args_NOnew, 'VRY_3':V_args_RY_3, 'VRY_4':V_args_RY_4},
            'f':{'G':f_args_G, 'no':f_args_no, 'fi':f_args_fi, 'VRY_2':f_args_RY_2, 'NOnew':f_args_NOnew, 'VRY_3':f_args_RY_3, 'VRY_4':f_args_RY_4},
            'ftype':{'G':'', 'no':'', 'fi':'', 'VRY_2':ftype, 'NOnew':'', 'VRY_3':'', 'VRY_4':ftype}
            }




################  lambdas  ################
lambdas_G = [252.0, 121.0**3.0, 972.0, 77.0**3.0] # T, s, mu, rho
Lambda_no = 831.0
kappa_5_no = 12.5
lambdas_no = [Lambda_no, 1.0/kappa_5_no*Lambda_no**3.0, Lambda_no, 1.0/kappa_5_no*Lambda_no**3.0] # [912.3685, 404.0779**3.0, 915.8167, 400.0341**3.0]
#lambdas_no = [921.149162573, 410.285737581**3.0, 1474.26455984, 345.083596761**3.0] # for new f_tanh type

#lambdas_RY = 4*[None]
#lambdas_RY[0] = 0.16249363/0.0219165731*155.0
#lambdas_RY[1] = 7.07901705547e-08/7.93413051e-07*lambdas_RY[0]**3.0
##chi2T2_scale = 1.0/(0.06449*lambdas_RY[1]/lambdas_RY[0]**3.0)
#chi2T2_scale = 1.0/(lambdas_RY[1]/lambdas_RY[0]**3.0)/f_tanh(0, *f_args_RY_2[1])
#lambdas_RY[2] = np.sqrt(chi2T2_scale*lambdas_RY[1]/lambdas_RY[0])
#lambdas_RY[3] = lambdas_RY[0]*lambdas_RY[1]/lambdas_RY[2]
lambdas_RY = [1148.06518555, 513.010128995**3.0, 1854.44816253, 429.575526095**3.0] # based on my fit (JK)


lambdas_dic = {'G':lambdas_G, 'no':lambdas_no, 'fi':lambdas_fi, 'VRY_2':lambdas_RY, 'NOnew':lambdas_NOnew, 'VRY_3':lambdas_RY_3, 'VRY_4':lambdas_RY_4}