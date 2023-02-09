import numpy as np
from scipy.interpolate import splev
from scipy.special import erf



## Gubser:
def V_I(t, *args):
    return -12.0*np.cosh(args[0]*t) + args[1]*t**2
def dV_I(t, *args):
    return -12.0*args[0]*np.sinh(args[0]*t) + 2.0*args[1]*t
def ddV_I(t, *args):
    return -12.0*args[0]**2.0*np.cosh(args[0]*t) + 2.0*args[1]
def dddV_I(t, *args):
    return -12.0*args[0]**3.0*np.sinh(args[0]*t)
def ddddV_I(t, *args):
    return -12.0*args[0]**4.0*np.cosh(args[0]*t)
def dddddV_I(t, *args):
    return -12.0*args[0]**5.0*np.sinh(args[0]*t)

def dlogV_I(t, *args):
    return dV_I(t, *args)/V_I(t, *args)


## Noronha:
def V_VI(t, *args):
    return -12.0*np.cosh(args[0]*t) + args[1]*t**2 + args[2]*t**4 + args[3]*t**6
def dV_VI(t, *args):
    return -12.0*args[0]*np.sinh(args[0]*t) + 2.0*args[1]*t + 4.0*args[2]*t**3 + 6.0*args[3]*t**5
def ddV_VI(t, *args):
    return -12.0*args[0]**2.0*np.cosh(args[0]*t) + 2.0*args[1] + 12.0*args[2]*t**2 + 30*args[3]*t**4
def dddV_VI(t, *args):
    return -12.0*args[0]**3.0*np.sinh(args[0]*t) + 24.0*args[2]*t + 120.0*args[3]*t**3
def ddddV_VI(t, *args):
    return -12.0*args[0]**4.0*np.cosh(args[0]*t) + 24.0*args[2] + 360.0*args[3]*t**2
def dddddV_VI(t, *args):
    return -12.0*args[0]**5.0*np.sinh(args[0]*t) + 720.0*args[3]*t

def dlogV_VI(t, *args):
    return dV_VI(t, *args)/V_VI(t, *args)


## derivative term for big phi:
def v_rpdb_pl(phi, *args):
    #print phi, args
    return args[0]*np.tanh(- args[1]*(args[2] - phi)) + args[3] + args[4]/np.cosh(-args[5]*(args[6] - phi))**2.0

def dv_rpdb_pl(phi, *args):
    return args[0]*args[1]/np.cosh(-args[1]*args[2] + args[1]*phi)**2.0 - 2.0*args[4]*args[5]*np.sinh(-args[5]*args[6] + args[5]*phi)/np.cosh(-args[5]*args[6] + args[5]*phi)**3.0

def ddv_rpdb_pl(phi, *args):
    return 4.0*args[4]*args[5]**2/np.cosh(-args[5]*args[6] + args[5]*phi)**2.0 - 2.0*args[0]*args[1]**2.0*np.sinh(-args[1]*args[2] + args[1]*phi)/np.cosh(-args[1]*args[2] + args[1]*phi)**3.0 - \
           6.0*args[4]*args[5]**2/np.cosh(-args[5]*args[6] + args[5]*phi)**4.0

def dddv_rpdb_pl(phi, *args):
    return 4.0*args[0]*args[1]**3.0/np.cosh(-args[1]*args[2] + args[1]*phi)**2.0 - \
            8.0*args[4]*args[5]**3.0*np.sinh(-args[5]*args[6] + args[5]*phi)/np.cosh(-args[5]*args[6] + args[5]*phi)**3.0 - 6.0*args[0]*args[1]**3.0/np.cosh(-args[1]*args[2] + args[1]*phi)**4.0 +\
            24.0*args[4]*args[5]**3.0*np.sinh(-args[5]*args[6] + args[5]*phi)/np.cosh(-args[5]*args[6] + args[5]*phi)**5.0

def ddddv_rpdb_pl(phi, *args):
    return 16.0*args[4]*args[5]**4/np.cosh(-args[5]*args[6] + args[5]*phi)**2.0 - 8.0*args[0]*args[1]**4.0*np.sinh(-args[1]*args[2] + args[1]*phi)/np.cosh(-args[1]*args[2] + args[1]*phi)**3.0 - \
    120.0*args[4]*args[5]**4.0/np.cosh(-args[5]*args[6] + args[5]*phi)**4.0 + 24.0*args[0]*args[1]**4.0*np.sinh(-args[1]*args[2] + args[1]*phi)/np.cosh(-args[1]*args[2] + args[1]*phi)**5.0 + \
                              120.0*args[4]*args[5]**4.0/np.cosh(-args[5]*args[6] + args[5]*phi)**6.0


## big phi term:
def V_rpdb_pl(phi, *args):
    return args[7]*np.cosh(- args[1]*(args[2] - phi))**(args[0]/args[1])*np.exp(args[3]*phi + args[4]*np.tanh(-args[5]*(args[6] - phi))/args[5])
def dV_rpdb_pl(phi, *args):
    return v_rpdb_pl(phi, *args)*V_rpdb_pl(phi, *args)
def ddV_rpdb_pl(phi, *args):
    return (v_rpdb_pl(phi, *args)**2.0 +  dv_rpdb_pl(phi, *args))*V_rpdb_pl(phi, *args)
def dddV_rpdb_pl(phi, *args):
    return (v_rpdb_pl(phi, *args)**3.0 + 3.0*v_rpdb_pl(phi, *args)*dv_rpdb_pl(phi, *args) + ddv_rpdb_pl(phi, *args))*V_rpdb_pl(phi, *args)
def ddddV_rpdb_pl(phi, *args):
    return (v_rpdb_pl(phi, *args)**4.0 + 6.0*v_rpdb_pl(phi, *args)**2.0*dv_rpdb_pl(phi, *args) + 3.0*dv_rpdb_pl(phi, *args)**2.0 +
            4.0*v_rpdb_pl(phi, *args)*ddv_rpdb_pl(phi, *args) + dddv_rpdb_pl(phi, *args))*V_rpdb_pl(phi, *args)
def dddddV_rpdb_pl(phi, *args):
    return (v_rpdb_pl(phi, *args)**5.0 + 10.0*v_rpdb_pl(phi, *args)**3.0*dv_rpdb_pl(phi, *args) + 15.0*v_rpdb_pl(phi, *args)*dv_rpdb_pl(phi, *args)**2.0
            + 10.0*(v_rpdb_pl(phi, *args)**2.0 + dv_rpdb_pl(phi, *args))*ddv_rpdb_pl(phi, *args)
            + 5.0*v_rpdb_pl(phi, *args)*dddv_rpdb_pl(phi, *args) + ddddv_rpdb_pl(phi, *args))*V_rpdb_pl(phi, *args)


## small phi term:
def V_13(phi, *args):
    return -12.0*np.exp(args[0]/2.0*phi**2.0 + args[1]/4.0*phi**4.0)

def v_13(phi, *args):
    return args[0]*phi + args[1]*phi**3.0
def dv_13(phi, *args):
    return args[0] + 3.0*args[1]*phi**2.0
def ddv_13(phi, *args):
    return 6.0*args[1]*phi
def dddv_13(phi, *args):
    return 6.0*args[1]
def ddddv_13(phi, *args):
    return 0

def dV_13(phi, *args):
    return v_13(phi, *args)*V_13(phi, *args)
def ddV_13(phi, *args):
    return (v_13(phi, *args)**2.0 +  dv_13(phi, *args))*V_13(phi, *args)
def dddV_13(phi, *args):
    return (v_13(phi, *args)**3.0 + 3.0*v_13(phi, *args)*dv_13(phi, *args) + ddv_13(phi, *args))*V_13(phi, *args)
def ddddV_13(phi, *args):
    return (v_13(phi, *args)**4.0 + 6.0*v_13(phi, *args)**2.0*dv_13(phi, *args) + 3.0*dv_13(phi, *args)**2.0 +
            4.0*v_13(phi, *args)*ddv_13(phi, *args) + dddv_13(phi, *args))*V_13(phi, *args)
def dddddV_13(phi, *args):
    return (v_13(phi, *args)**5.0 + 10.0*v_13(phi, *args)**3.0*dv_13(phi, *args) + 15.0*v_13(phi, *args)*dv_13(phi, *args)**2.0
            + 10.0*(v_13(phi, *args)**2.0 + dv_13(phi, *args))*ddv_13(phi, *args)
            + 5.0*v_13(phi, *args)*dddv_13(phi, *args) + ddddv_13(phi, *args))*V_13(phi, *args)


## VRY_2:
def V_rpdb_pl_s(phi, *args):
    phih_m = args[0]
    if phi < phih_m:
        return V_13(phi, *args[1:3])
    elif phi >= phih_m:
        return V_rpdb_pl(phi, *args[3:])
def dV_rpdb_pl_s(phi, *args):
    phih_m = args[0]
    if phi < phih_m:
        return dV_13(phi, *args[1:3])
    elif phi >= phih_m:
        return dV_rpdb_pl(phi, *args[3:])
def ddV_rpdb_pl_s(phi, *args):
    phih_m = args[0]
    if phi < phih_m:
        return ddV_13(phi, *args[1:3])
    elif phi >= phih_m:
        return ddV_rpdb_pl(phi, *args[3:])
def dddV_rpdb_pl_s(phi, *args):
    phih_m = args[0]
    if phi < phih_m:
        return dddV_13(phi, *args[1:3])
    elif phi >= phih_m:
        return dddV_rpdb_pl(phi, *args[3:])
def ddddV_rpdb_pl_s(phi, *args):
    phih_m = args[0]
    if phi < phih_m:
        return ddddV_13(phi, *args[1:3])
    elif phi >= phih_m:
        return ddddV_rpdb_pl(phi, *args[3:])
def dddddV_rpdb_pl_s(phi, *args):
    phih_m = args[0]
    if phi < phih_m:
        return dddddV_13(phi, *args[1:3])
    elif phi >= phih_m:
        return dddddV_rpdb_pl(phi, *args[3:])

def dlogV_rpdb_pl_s(phi, *args):
    dV_rpdb_pl_s(phi, *args)/V_rpdb_pl_s(phi, *args)


## VRY_2 with linear end term:
r = 3.12833
m = -23.6905
n = 35.377
c0 = -39.1302
c1 = 23.9433
c2 = -7.61329
def V_lin(phi, *args):
    phih_m = args[0]
    if phi < phih_m:
        return V_13(phi, *args[1:3])
    elif phi >= phih_m and phi<=r:
        return V_rpdb_pl(phi, *args[3:])
    else:
        return c0 + c1*phi + c2*phi**2 # m*phi + n
def dV_lin(phi, *args):
    phih_m = args[0]
    if phi < phih_m:
        return dV_13(phi, *args[1:3])
    elif phi >= phih_m and phi<=r:
        return dV_rpdb_pl(phi, *args[3:])
    else:
        return c1 + 2*c2*phi # m
def ddV_lin(phi, *args):
    phih_m = args[0]
    if phi < phih_m:
        return ddV_13(phi, *args[1:3])
    elif phi >= phih_m and phi<=r:
        return ddV_rpdb_pl(phi, *args[3:])
    else:
        return 2*c2 # 0.0
def dddV_lin(phi, *args):
    phih_m = args[0]
    if phi < phih_m:
        return dddV_13(phi, *args[1:3])
    elif phi >= phih_m and phi<=r:
        return dddV_rpdb_pl(phi, *args[3:])
    else:
        return 0.0
def ddddV_lin(phi, *args):
    phih_m = args[0]
    if phi < phih_m:
        return ddddV_13(phi, *args[1:3])
    elif phi >= phih_m and phi<=r:
        return ddddV_rpdb_pl(phi, *args[3:])
    else:
        return 0.0
def dddddV_lin(phi, *args):
    phih_m = args[0]
    if phi < phih_m:
        return dddddV_13(phi, *args[1:3])
    elif phi >= phih_m and phi<=r:
        return dddddV_rpdb_pl(phi, *args[3:])
    else:
        return 0.0

def dlogV_lin(phi, *args):
    dV_lin(phi, *args)/V_lin(phi, *args)


## additional exp term:
def V_add(phi, *args):
    a8,a9,a10 = args[8],args[9],args[10]
    return  np.exp(a9*(phi-a8)) * (np.tanh(a10*(phi-a8))+1.0)/2.0
def dV_add(phi, *args):
    a8,a9,a10 = args[8],args[9],args[10]
    return 1/2.0 *a10 *np.exp(a9*(-a8+phi)) /np.cosh(a10*(-a8+phi))**2 + 1/2.0* a9 *np.exp(a9*(-a8+phi))* (1.0+np.tanh(a10*(-a8+phi)))
def ddV_add(phi, *args):
    a8,a9,a10 = args[8],args[9],args[10]
    return a10* a9 *np.exp(a9*(-a8+phi)) /np.cosh(a10*(-a8+phi))**2 - a10**2 *np.exp(a9*(-a8+phi)) /np.cosh(a10*(-a8+phi))**2 *np.tanh(a10*(-a8+phi)) + 1/2.0* a9**2 *np.exp(a9*(-a8+phi)) *(1.0+np.tanh(a10*(-a8+phi)))
def dddV_add(phi, *args):
    a8,a9,a10 = args[8],args[9],args[10]
    return 3/2.0 *a10 *a9**2 *np.exp(a9*(-a8+phi)) /np.cosh(a10*(-a8+phi))**2 - 3.0* a10**2 *a9 *np.exp(a9*(-a8+phi)) /np.cosh(a10*(-a8+phi))**2 *np.tanh(a10*(-a8+phi)) + 1/2.0* a9**3 *np.exp(a9*(-a8+phi)) *(1.0+np.tanh(a10*(-a8+phi))) + 1/2.0 *np.exp(a9*(-a8+phi)) *(-2.0* a10**3 /np.cosh(a10*(-a8+phi))**4 + 4.0* a10**3 /np.cosh(a10*(-a8+phi))**2 *np.tanh(a10*(-a8+phi))**2)
def ddddV_add(phi, *args):
    a8,a9,a10 = args[8],args[9],args[10]
    return 2.0* a10* a9**3 *np.exp(a9*(-a8+phi)) /np.cosh(a10*(-a8+phi))**2 - 6.0* a10**2 *a9**2 *np.exp(a9*(-a8+phi)) /np.cosh(a10*(-a8+phi))**2 *np.tanh(a10*(-a8+phi)) + 1/2.0* a9**4 *np.exp(a9*(-a8+phi))* (1.0+np.tanh(a10*(-a8+phi))) + 2.0* a9 *np.exp(a9*(-a8+phi))* (-2.0* a10**3 /np.cosh(a10*(-a8+phi))**4 + 4.0* a10**3 /np.cosh(a10*(-a8+phi))**2 *np.tanh(a10*(-a8+phi))**2) + 1/2.0 *np.exp(a9*(-a8+phi)) *(16.0* a10**4 /np.cosh(a10*(-a8+phi))**4 *np.tanh(a10*(-a8+phi)) - 8.0* a10**4 /np.cosh(a10*(-a8+phi))**2 *np.tanh(a10*(-a8+phi))**3)
def dddddV_add(phi, *args):
    a8,a9,a10 = args[8],args[9],args[10]
    return 5/2.0 *a10* a9**4 *np.exp(a9*(-a8+phi)) /np.cosh(a10*(-a8+phi))**2 - 10.0* a10**2 *a9**3 *np.exp(a9*(-a8+phi)) /np.cosh(a10*(-a8+phi))**2 *np.tanh(a10*(-a8+phi)) + 1/2.0* a9**5 *np.exp(a9*(-a8+phi))* (1.0+np.tanh(a10*(-a8+phi))) + 5.0* a9**2 *np.exp(a9*(-a8+phi))* (-2.0* a10**3 /np.cosh(a10*(-a8+phi))**4 + 4.0* a10**3 /np.cosh(a10*(-a8+phi))**2 *np.tanh(a10*(-a8+phi))**2) + 5/2.0* a9 *np.exp(a9*(-a8+phi)) *(16.0* a10**4 /np.cosh(a10*(-a8+phi))**4 *np.tanh(a10*(-a8+phi)) - 8.0* a10**4 /np.cosh(a10*(-a8+phi))**2 *np.tanh(a10*(-a8+phi))**3) + 1/2.0 *np.exp(a9*(-a8+phi))* (16.0* a10**5 /np.cosh(a10*(-a8+phi))**6 - 88.0* a10**5 /np.cosh(a10*(-a8+phi))**4 *np.tanh(a10*(-a8+phi))**2 + 16.0* a10**5 /np.cosh(a10*(-a8+phi))**2 *np.tanh(a10*(-a8+phi))**4)


## VRY_2 with exp term:
def V_exp(phi, *args):
    phih_m = args[0]
    if phi < phih_m:
        return V_13(phi, *args[1:3])
    elif phi >= phih_m:
        return V_rpdb_pl(phi, *args[3:]) - V_add(phi, *args[3:])
def dV_exp(phi, *args):
    phih_m = args[0]
    if phi < phih_m:
        return dV_13(phi, *args[1:3])
    elif phi >= phih_m:
        return dV_rpdb_pl(phi, *args[3:]) - dV_add(phi, *args[3:])
def ddV_exp(phi, *args):
    phih_m = args[0]
    if phi < phih_m:
        return ddV_13(phi, *args[1:3])
    elif phi >= phih_m:
        return ddV_rpdb_pl(phi, *args[3:]) - ddV_add(phi, *args[3:])
def dddV_exp(phi, *args):
    phih_m = args[0]
    if phi < phih_m:
        return dddV_13(phi, *args[1:3])
    elif phi >= phih_m:
        return dddV_rpdb_pl(phi, *args[3:]) - dddV_add(phi, *args[3:])
def ddddV_exp(phi, *args):
    phih_m = args[0]
    if phi < phih_m:
        return ddddV_13(phi, *args[1:3])
    elif phi >= phih_m:
        return ddddV_rpdb_pl(phi, *args[3:]) - ddddV_add(phi, *args[3:])
def dddddV_exp(phi, *args):
    phih_m = args[0]
    if phi < phih_m:
        return dddddV_13(phi, *args[1:3])
    elif phi >= phih_m:
        return dddddV_rpdb_pl(phi, *args[3:]) - dddddV_add(phi, *args[3:])

def dlogV_exp(phi, *args):
    dV_exp(phi, *args)/V_exp(phi, *args)





##=======================================================================================================================================================
Vs = {'V_I':V_I, 'V_VI':V_VI, 'V_rpdb_pl_s':V_rpdb_pl_s, 'V_exp':V_exp, 'V_lin':V_lin}
dVs = {'V_I':dV_I, 'V_VI':dV_VI, 'V_rpdb_pl_s':dV_rpdb_pl_s, 'V_exp':dV_exp, 'V_lin':dV_lin}
ddVs = {'V_I':ddV_I, 'V_VI':ddV_VI, 'V_rpdb_pl_s':ddV_rpdb_pl_s, 'V_exp':ddV_exp, 'V_lin':ddV_lin}
dddVs = {'V_I':dddV_I, 'V_VI':dddV_VI, 'V_rpdb_pl_s':dddV_rpdb_pl_s, 'V_exp':dddV_exp, 'V_lin':dddV_lin}
ddddVs = {'V_I':ddddV_I, 'V_VI':ddddV_VI, 'V_rpdb_pl_s':ddddV_rpdb_pl_s, 'V_exp':ddddV_exp, 'V_lin':ddddV_lin}
dddddVs = {'V_I':dddddV_I, 'V_VI':dddddV_VI, 'V_rpdb_pl_s':dddddV_rpdb_pl_s, 'V_exp':dddddV_exp, 'V_lin':dddddV_lin}
dlogVs = {'V_I':dlogV_I, 'V_VI':dlogV_VI, 'V_rpdb_pl_s':dlogV_rpdb_pl_s, 'V_exp':dlogV_exp, 'V_lin':dlogV_lin}
