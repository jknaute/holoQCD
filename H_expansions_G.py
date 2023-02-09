import numpy as np

from Vtypes import Vs, dVs, ddVs, dddVs, ddddVs, dddddVs, dlogVs
from ftypes import fs, dfs, ddfs, dddfs, ddddfs, dddddfs, dlogfs
L = 1.0

def V(phi, *args):
    return Vs[args[0]](phi, *args[1])

def V_1(phi, *args):
    return dVs[args[0]](phi, *args[1])

def V_2(phi, *args):
    return ddVs[args[0]](phi, *args[1])

def V_3(phi, *args):
    return dddVs[args[0]](phi, *args[1])

def V_4(phi, *args):
    return ddddVs[args[0]](phi, *args[1])

def V_5(phi, *args):
    return dddddVs[args[0]](phi, *args[1])

def f(phi, *args):
    return fs[args[0]](phi, *args[1])

def f_1(phi, *args):
    return dfs[args[0]](phi, *args[1])

def f_2(phi, *args):
    return ddfs[args[0]](phi, *args[1])

def f_3(phi, *args):
    return dddfs[args[0]](phi, *args[1])

def f_4(phi, *args):
    return ddddfs[args[0]](phi, *args[1])

def f_5(phi, *args):
    return dddddfs[args[0]](phi, *args[1])

def hex_coeffs(max_order, phi_0, Phi_1, V_args, f_args):
    V0 = V(phi_0, *V_args)
    V1 = V_1(phi_0, *V_args)
    V2 = V_2(phi_0, *V_args)
    V3 = V_3(phi_0, *V_args)
    V4 = V_4(phi_0, *V_args)
    V5 = V_5(phi_0, *V_args)

    f0 = f(phi_0, *f_args)
    f1 = f_1(phi_0, *f_args)
    f2 = f_2(phi_0, *f_args)
    f3 = f_3(phi_0, *f_args)
    f4 = f_4(phi_0, *f_args)
    f5 = f_5(phi_0, *f_args)

    A_0 = 0
    h_0 = 0
    Phi_0 = 0

    A_1 = -1.0/3.0*L*(1.0/2.0*f0*Phi_1**2.0 + V0)
    phi_1 = L*(V1 - 1.0/2.0*f1*Phi_1**2.0)
    h_1 = 1.0/L

    A_2 = - 1.0/12.0*phi_1**2.0
    Phi_2 = - Phi_1/(2.0*f0)*(2.0*A_1*f0 + f1*phi_1)
    h_2 = 5.0/6.0*Phi_1**2.0*f0 + 2.0/3.0*V0
    phi_2 = 1.0/4.0*L*Phi_1*f1*(A_1*Phi_1 - 2.0*Phi_2) - 1.0/8.0*phi_1*(8.0*A_1 + L*(Phi_1**2.0*f2 - 2.0*V2 + 4.0*h_2))

    A_3 = - 1.0/9.0*phi_1*phi_2
    Phi_3 = - 1.0/(6.0*f0**2.0)*(4.0*A_2*Phi_1*f0**2.0 + 4.0*A_1*Phi_2*f0**2.0 + 2.0*Phi_2*f0*f1*phi_1 + 2.0*Phi_1*f0*f1*phi_2 - (Phi_1*f1**2.0 - Phi_1*f0*f2)*phi_1**2.0)
    h_3 = 1.0/6.0*(L*Phi_1**2.0*f1*phi_1 + 2.0*(2.0*Phi_1*Phi_2*f0 - (Phi_1**2.0*f0 + 4.0*h_2)*A_1)*L - 8.0*A_2)/L
    phi_3 = -1.0/36.0*(Phi_1**2.0*f3 - 2.0*V3)*L*phi_1**2.0 - 1.0/9.0*(A_1**2.0*Phi_1**2.0*f1 - A_2*Phi_1**2.0*f1 - 4.0*A_1*Phi_1*Phi_2*f1 + (2.0*Phi_2**2.0 + 3.0*Phi_1*Phi_3)*f1)*L - 1.0/9.0*((2.0*Phi_1*Phi_2*f2 - (Phi_1**2.0*f2 - 4.0*h_2)*A_1 + 3.0*h_3)*L + 8.0*A_2)*phi_1 - 1.0/18.0*((Phi_1**2.0*f2 - 2.0*V2 + 12.0*h_2)*L + 16.0*A_1)*phi_2

    A_4 = -1.0/18.0*phi_2**2.0 - 1.0/12.0*phi_1*phi_3
    Phi_4 = -1.0/24.0*(12.0*A_3*Phi_1*f0**3.0 + 16.0*A_2*Phi_2*f0**3.0 + 12.0*A_1*Phi_3*f0**3.0 + 6.0*Phi_3*f0**2.0*f1*phi_1 + 6.0*Phi_1*f0**2.0*f1*phi_3 + (2.0*Phi_1*f1**3.0 - 3.0*Phi_1*f0*f1*f2 + Phi_1*f0**2.0*f3)*phi_1**3.0 - 4.0*(Phi_2*f0*f1**2.0 - Phi_2*f0**2.0*f2)*phi_1**2.0 + 2.0*(4.0*Phi_2*f0**2.0*f1 - 3.0*(Phi_1*f0*f1**2.0 - Phi_1*f0**2.0*f2)*phi_1)*phi_2)/f0**3.0
    h_4 = 1.0/24.0*((4.0*A_1**2.0*Phi_1**2.0*f0 + Phi_1**2.0*f2*phi_1**2.0 + 8.0*Phi_1*Phi_2*f1*phi_1 + 2.0*Phi_1**2.0*f1*phi_2 - 4.0*(Phi_1**2.0*f1*phi_1 + 4.0*Phi_1*Phi_2*f0 + 6.0*h_3)*A_1 - 4.0*(Phi_1**2.0*f0 + 8.0*h_2)*A_2 + 4.0*(2.0*Phi_2**2.0 + 3.0*Phi_1*Phi_3)*f0)*L - 24.0*A_3)/L
    phi_4 = 1.0/96.0*L*V4*phi_1**3.0 + 1.0/16.0*L*V3*phi_1*phi_2 + 1.0/16.0*L*V2*phi_3 + 1.0/192.0*(8.0*A_1**3.0*Phi_1**2.0*f1 - Phi_1**2.0*f4*phi_1**3.0 - 12.0*Phi_1*Phi_2*f3*phi_1**2.0 + 12.0*A_3*Phi_1**2.0*f1 - 12.0*(Phi_1**2.0*f2*phi_1 + 4.0*Phi_1*Phi_2*f1)*A_1**2.0 + 6.0*(Phi_1**2.0*f3*phi_1**2.0 + 4.0*(2.0*Phi_2**2.0 + 3.0*Phi_1*Phi_3)*f1 + 8.0*(Phi_1*Phi_2*f2 - h_3)*phi_1 + 2.0*(Phi_1**2.0*f2 - 8.0*h_2)*phi_2)*A_1 - 12.0*(2.0*A_1*Phi_1**2.0*f1 - 4.0*Phi_1*Phi_2*f1 - (Phi_1**2.0*f2 - 8.0*h_2)*phi_1)*A_2 - 24.0*(3.0*Phi_2*Phi_3 + 2.0*Phi_1*Phi_4)*f1 - 12.0*((2.0*Phi_2**2.0 + 3.0*Phi_1*Phi_3)*f2 + 4.0*h_4)*phi_1 - 6.0*(Phi_1**2.0*f3*phi_1 + 4.0*Phi_1*Phi_2*f2 + 16.0*h_3)*phi_2 - 6.0*(Phi_1**2.0*f2 + 24.0*h_2)*phi_3)*L - 3.0/4.0*A_3*phi_1 - A_2*phi_2 - 3.0/4.0*A_1*phi_3

    A_5 = -1.0/10.0*phi_2*phi_3 - 1.0/15.0*phi_1*phi_4
    Phi_5 = -1.0/120.0*(48.0*A_4*Phi_1*f0**4.0 + 72.0*A_3*Phi_2*f0**4.0 + 72.0*A_2*Phi_3*f0**4.0 + 48.0*A_1*Phi_4*f0**4.0 + 24.0*Phi_4*f0**3.0*f1*phi_1 + 24.0*Phi_1*f0**3.0*f1*phi_4 - (6.0*Phi_1*f1**4.0 - 12.0*Phi_1*f0*f1**2.0*f2 + 3.0*Phi_1*f0**2.0*f2**2.0 + 4.0*Phi_1*f0**2.0*f1*f3 - Phi_1*f0**3.0*f4)*phi_1**4.0 + 6.0*(2.0*Phi_2*f0*f1**3.0 - 3.0*Phi_2*f0**2.0*f1*f2 + Phi_2*f0**3.0*f3)*phi_1**3.0 - 18.0*(Phi_3*f0**2.0*f1**2.0 - Phi_3*f0**3.0*f2)*phi_1**2.0 - 12.0*(Phi_1*f0**2.0*f1**2.0 - Phi_1*f0**3.0*f2)*phi_2**2.0 + 12.0*(3.0*Phi_3*f0**3.0*f1 + (2.0*Phi_1*f0*f1**3.0 - 3.0*Phi_1*f0**2.0*f1*f2 + Phi_1*f0**3.0*f3)*phi_1**2.0 - 3.0*(Phi_2*f0**2.0*f1**2.0 - Phi_2*f0**3.0*f2)*phi_1)*phi_2 + 12.0*(3.0*Phi_2*f0**3.0*f1 - 2.0*(Phi_1*f0**2.0*f1**2.0 - Phi_1*f0**3.0*f2)*phi_1)*phi_3)/f0**4.0
    h_5 = -1.0/120.0*((8.0*A_1**3.0*Phi_1**2.0*f0 - Phi_1**2.0*f3*phi_1**3.0 - 12.0*Phi_1*Phi_2*f2*phi_1**2.0 - 6.0*Phi_1**2.0*f1*phi_3 - 12.0*(Phi_1**2.0*f1*phi_1 + 4.0*Phi_1*Phi_2*f0)*A_1**2.0 - 12.0*(2.0*Phi_2**2.0 + 3.0*Phi_1*Phi_3)*f1*phi_1 + 6.0*(Phi_1**2.0*f2*phi_1**2.0 + 8.0*Phi_1*Phi_2*f1*phi_1 + 2.0*Phi_1**2.0*f1*phi_2 + 4.0*(2.0*Phi_2**2.0 + 3.0*Phi_1*Phi_3)*f0 + 16.0*h_4)*A_1 - 12.0*(2.0*A_1*Phi_1**2.0*f0 - Phi_1**2.0*f1*phi_1 - 4.0*Phi_1*Phi_2*f0 - 12.0*h_3)*A_2 + 12.0*(Phi_1**2.0*f0 + 12.0*h_2)*A_3 - 24.0*(3.0*Phi_2*Phi_3 + 2.0*Phi_1*Phi_4)*f0 - 6.0*(Phi_1**2.0*f2*phi_1 + 4.0*Phi_1*Phi_2*f1)*phi_2)*L + 96.0*A_4)/L
    phi_5 =  1.0/600.0*L*V5*phi_1**4 + 1.0/50.0*L*V4*phi_1**2*phi_2 + 1.0/50.0*(phi_2**2.0 + 2.0*phi_1*phi_3)*L*V3 + 1.0/25.0*L*V2*phi_4 - 1.0/1200.0*(16.0*A_1**4.0*Phi_1**2.0*f1 + Phi_1**2.0*f5*phi_1**4.0 + 16.0*Phi_1*Phi_2*f4*phi_1**3.0 + 48.0*A_2**2.0*Phi_1**2.0*f1 + 12.0*Phi_1**2.0*f3*phi_2**2.0 - 32.0*(Phi_1**2.0*f2*phi_1 + 4.0*Phi_1*Phi_2*f1)*A_1**3.0 - 48.0*A_4*Phi_1**2.0*f1 + 24.0*(2.0*Phi_2**2.0 + 3.0*Phi_1*Phi_3)*f3*phi_1**2.0 + 24.0*(Phi_1**2.0*f3*phi_1**2.0 + 8.0*Phi_1*Phi_2*f2*phi_1 + 2.0*Phi_1**2.0*f2*phi_2 + 4.0*(2.0*Phi_2**2.0 + 3.0*Phi_1*Phi_3)*f1)*A_1**2.0 - 8.0*(Phi_1**2*f4*phi_1**3.0 + 12.0*Phi_1*Phi_2*f3*phi_1**2.0 + 24.0*(3.0*Phi_2*Phi_3 + 2.0*Phi_1*Phi_4)*f1 + 12.0*((2.0*Phi_2**2.0 + 3.0*Phi_1*Phi_3)*f2 - 2.0*h_4)*phi_1 + 6.0*(Phi_1**2.0*f3*phi_1 + 4.0*Phi_1*Phi_2*f2 - 8.0*h_3)*phi_2 + 6.0*(Phi_1**2.0*f2 - 12.0*h_2)*phi_3)*A_1 - 24.0*(4.0*A_1**2.0*Phi_1**2.0*f1 + Phi_1**2.0*f3*phi_1**2.0 - 4.0*(Phi_1**2.0*f2*phi_1 + 4.0*Phi_1*Phi_2*f1)*A_1 + 4.0*(2.0*Phi_2**2.0 + 3.0*Phi_1*Phi_3)*f1 + 8.0*(Phi_1*Phi_2*f2 - 2.0*h_3)*phi_1 + 2.0*(Phi_1**2.0*f2 - 16.0*h_2)*phi_2)*A_2 + 48.0*(2.0*A_1*Phi_1**2.0*f1 - 4.0*Phi_1*Phi_2*f1 - (Phi_1**2.0*f2 - 12.0*h_2)*phi_1)*A_3 + 24.0*(9.0*Phi_3**2.0 + 16.0*Phi_2*Phi_4 + 10.0*Phi_1*Phi_5)*f1 + 48.0*(2.0*(3.0*Phi_2*Phi_3 + 2.0*Phi_1*Phi_4)*f2 + 5.0*h_5)*phi_1 + 12.0*(Phi_1**2.0*f4*phi_1**2.0 + 8.0*Phi_1*Phi_2*f3*phi_1 + 4.0*(2*Phi_2**2.0 + 3.0*Phi_1*Phi_3)*f2 + 40.0*h_4)*phi_2 + 24.0*(Phi_1**2.0*f3*phi_1 + 4.0*Phi_1*Phi_2*f2 + 30.0*h_3)*phi_3 + 24.0*(Phi_1**2.0*f2 + 40.0*h_2)*phi_4)*L - 16.0/25.0*A_4*phi_1 - 24.0/25.0*A_3*phi_2 - 24.0/25.0*A_2*phi_3 - 16.0/25.0*A_1*phi_4

    A_coeffs = [A_0, A_1, A_2, A_3, A_4, A_5][:max_order + 1]
    h_coeffs = [h_0, h_1, h_2, h_3, h_4, h_5][:max_order + 1]
    phi_coeffs = [phi_0, phi_1, phi_2, phi_3, phi_4, phi_5][:max_order + 1]
    Phi_coeffs = [Phi_0, Phi_1, Phi_2, Phi_3, Phi_4, Phi_5][:max_order + 1]

    dA_coeffs = [A_1, 2.0*A_2, 3.0*A_3, 4.0*A_4, 5.0*A_5][:max_order]
    dh_coeffs = [h_1, 2.0*h_2, 3.0*h_3, 4.0*h_4, 5.0*h_5][:max_order]
    dphi_coeffs = [phi_1, 2.0*phi_2, 3.0*phi_3, 4.0*phi_4, 5.0*phi_5][:max_order]
    dPhi_coeffs = [Phi_1, 2.0*Phi_2, 3.0*Phi_3, 4.0*Phi_4, 5.0*Phi_5][:max_order]

    return {'A':A_coeffs, 'h':h_coeffs, 'phi':phi_coeffs, 'Phi':Phi_coeffs, 'dA':dA_coeffs, 'dh':dh_coeffs, 'dphi':dphi_coeffs, 'dPhi':dPhi_coeffs}


def get_horizon_expansions(rho, order, phi_0, Phi_1, V_args, f_args): # rho := r - r_H
    funcs_list = ['A', 'dA', 'phi', 'dphi', 'h', 'dh', 'Phi', 'dPhi']
    hex_coeffs_vals = hex_coeffs(order, phi_0, Phi_1, V_args, f_args)
    hexs_arr = np.zeros(len(funcs_list))
    hexs_dic = {}
    j = 0
    for func in funcs_list:
        func_hex = 0
        max_order = order

        if func[0] == 'd':
            max_order -= 1
        for i in range(0, max_order + 1):
            func_hex = func_hex + hex_coeffs_vals[func][i]*rho**i
            #print "horizon exp: ", func, i, hex_coeffs_vals[func][i], hex_coeffs_vals[func][i]*rho**i, func_hex

        #print func, func_hex
        hexs_arr[j] = func_hex
        hexs_dic.update({func:func_hex})
        j += 1

    return [hexs_arr, hexs_dic]


