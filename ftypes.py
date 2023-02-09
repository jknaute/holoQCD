import numpy as np
from scipy.interpolate import splev
from scipy.special import erf


## Gubser:
def f_I(t, *args):
    return args[0]/np.cosh(args[1]*(t - args[2]))
def df_I(t, *args):
    return - args[0]*args[1]*np.sinh(args[1]*(t - args[2]))/np.cosh(args[1]*(t - args[2]))**2.0
def ddf_I(t, *args):
    return args[0]*args[1]**2.0*(np.sinh(args[1]*(t - args[2]))**2.0 - 1.0)/np.cosh(args[1]*(t - args[2]))**3.0
def dddf_I(t, *args):
    return - args[0]*args[1]**3.0*np.sinh(args[1]*(t - args[2]))*(np.sinh(args[1]*(t - args[2]))**2.0 - 5.0)/np.cosh(args[1]*(t - args[2]))**4.0
def ddddf_I(t, *args):
    return args[0]*args[1]**4.0*(np.sinh(args[1]*(t - args[2]))**4.0 - 18.0*np.sinh(args[1]*(t - args[2]))**2.0 + 5.0)/np.cosh(args[1]*(t - args[2]))**5.0
def dddddf_I(t, *args):
    return - args[0]*args[1]**5.0*np.sinh(args[1]*(t - args[2]))*(np.sinh(args[1]*(t - args[2]))**4.0 - 58.0*np.sinh(args[1]*(t - args[2]))**2.0  + 61.0)/np.cosh(args[1]*(t - args[2]))**6.0

def dlogf_I(t, *args):
    return df_I(t, *args)/f_I(t, *args)


## Noronha:
def f_no(t, *args):
    return args[0]/np.cosh(args[1]*(t - args[2])) + args[3]*np.exp(args[4]*t)
def df_no(t, *args):
    return - args[0]*args[1]*np.sinh(args[1]*(t - args[2]))/np.cosh(args[1]*(t - args[2]))**2.0 + args[3]*args[4]*np.exp(args[4]*t)
def ddf_no(t, *args):
    return args[0]*args[1]**2.0*(np.sinh(args[1]*(t - args[2]))**2.0 - 1.0)/np.cosh(args[1]*(t - args[2]))**3.0 + args[3]*args[4]**2.0*np.exp(args[4]*t)
def dddf_no(t, *args):
    return - args[0]*args[1]**3.0*np.sinh(args[1]*(t - args[2]))*(np.sinh(args[1]*(t - args[2]))**2.0 - 5.0)/np.cosh(args[1]*(t - args[2]))**4.0 + args[3]*args[4]**3.0*np.exp(args[4]*t)
def ddddf_no(t, *args):
    return args[0]*args[1]**4.0*(np.sinh(args[1]*(t - args[2]))**4.0 - 18.0*np.sinh(args[1]*(t - args[2]))**2.0 + 5.0)/np.cosh(args[1]*(t - args[2]))**5.0 + args[3]*args[4]**4.0*np.exp(args[4]*t)
def dddddf_no(t, *args):
    return - args[0]*args[1]**5.0*np.sinh(args[1]*(t - args[2]))*(np.sinh(args[1]*(t - args[2]))**4.0 - 58.0*np.sinh(args[1]*(t - args[2]))**2.0  + 61.0)/np.cosh(args[1]*(t - args[2]))**6.0 + args[3]*args[4]**5.0*np.exp(args[4]*t)

def dlogf_no(t, *args):
    return df_no(t, *args)/f_no(t, *args)


## Finazzo:
def f_fi(phi, *args):
    nrm, c2, c1, c0 = args[0], args[1], args[2], args[3]
    return nrm/np.cosh(c2*phi**2.0 + c1*phi + c0)
def df_fi(phi, *args):
    nrm, c2, c1, c0 = args[0], args[1], args[2], args[3]
    return -nrm *(c1+2*c2*phi) /np.cosh(c0+c1*phi+c2*phi**2) *np.tanh(c0+c1*phi+c2*phi**2)
def ddf_fi(phi, *args):
    nrm, c2, c1, c0 = args[0], args[1], args[2], args[3]
    return nrm *(-(c1+2*c2*phi)**2 /np.cosh(c0+c1*phi+c2*phi**2)**3 - 2*c2 /np.cosh(c0+c1*phi+c2*phi**2) *np.tanh(c0+c1*phi+c2*phi**2)+(c1+2*c2*phi)**2 /np.cosh(c0+c1*phi+c2*phi**2) *np.tanh(c0+c1*phi+c2*phi**2)**2)
def dddf_fi(phi, *args):
    nrm, c2, c1, c0 = args[0], args[1], args[2], args[3]
    return nrm *(-6 *c2* (c1+2*c2*phi) /np.cosh(c0+c1*phi+c2*phi**2)**3+5 *(c1+2*c2*phi)**3 /np.cosh(c0+c1*phi+c2*phi**2)**3 *np.tanh(c0+c1*phi+c2*phi**2)+6* c2* (c1+2*c2*phi) /np.cosh(c0+c1*phi+c2*phi**2) *np.tanh(c0+c1*phi+c2*phi**2)**2-(c1+2*c2*phi)**3 /np.cosh(c0+c1*phi+c2*phi**2) *np.tanh(c0+c1*phi+c2*phi**2)**3)
def ddddf_fi(phi, *args):
    nrm, c2, c1, c0 = args[0], args[1], args[2], args[3]
    return nrm *(-12 *c2**2 /np.cosh(c0+c1*phi+c2*phi**2)**3 + 5*(c1+2*c2*phi)**4 /np.cosh(c0+c1*phi+c2*phi**2)**5 + 60*c2*(c1+2*c2*phi)**2 /np.cosh(c0+c1*phi+c2*phi**2)**3 *np.tanh(c0+c1*phi+c2*phi**2)+12 *c2**2 /np.cosh(c0+c1*phi+c2*phi**2) *np.tanh(c0+c1*phi+c2*phi**2)**2-18 *(c1+2*c2*phi)**4 /np.cosh(c0+c1*phi+c2*phi**2)**3 *np.tanh(c0+c1*phi+c2*phi**2)**2-12 *c2* (c1+2*c2*phi)**2 /np.cosh(c0+c1*phi+c2*phi**2) *np.tanh(c0+c1*phi+c2*phi**2)**3+(c1+2*c2*phi)**4 /np.cosh(c0+c1*phi+c2*phi**2) *np.tanh(c0+c1*phi+c2*phi**2)**4)
def dddddf_fi(phi, *args):
    nrm, c2, c1, c0 = args[0], args[1], args[2], args[3]
    return nrm *(100* c2* (c1+2*c2*phi)**3 /np.cosh(c0+c1*phi+c2*phi**2)**5+300 *c2**2* (c1+2*c2*phi) /np.cosh(c0+c1*phi+c2*phi**2)**3 *np.tanh(c0+c1*phi+c2*phi**2)-61* (c1+2*c2*phi)**5 /np.cosh(c0+c1*phi+c2*phi**2)**5 *np.tanh(c0+c1*phi+c2*phi**2)-360* c2* (c1+2*c2*phi)**3 /np.cosh(c0+c1*phi+c2*phi**2)**3 *np.tanh(c0+c1*phi+c2*phi**2)**2-60* c2**2 *(c1+2*c2*phi) /np.cosh(c0+c1*phi+c2*phi**2) *np.tanh(c0+c1*phi+c2*phi**2)**3+58* (c1+2*c2*phi)**5 /np.cosh(c0+c1*phi+c2*phi**2)**3 *np.tanh(c0+c1*phi+c2*phi**2)**3+20* c2* (c1+2*c2*phi)**3 /np.cosh(c0+c1*phi+c2*phi**2) *np.tanh(c0+c1*phi+c2*phi**2)**4-(c1+2*c2*phi)**5 /np.cosh(c0+c1*phi+c2*phi**2) *np.tanh(c0+c1*phi+c2*phi**2)**5)

def dlogf_fi(phi, *args):
    return df_fi(phi, *args)/f_fi(phi, *args)


## newest Noronha with CEP:
def f_sech(phi, *args):
    c1, c2, c3, c4 = args[0], args[1], args[2], args[3]
    return (c3 /np.cosh(c4*phi)+1/np.cosh(c1*phi+c2*phi**2))/(1+c3)
def df_sech(phi, *args):
    c1, c2, c3, c4 = args[0], args[1], args[2], args[3]
    return (-c3 *c4 /np.cosh(c4*phi) *np.tanh(c4*phi)-(c1+2*c2*phi) /np.cosh(c1*phi+c2*phi**2) *np.tanh(c1*phi+c2*phi**2))/(1+c3)
def ddf_sech(phi, *args):
    c1, c2, c3, c4 = args[0], args[1], args[2], args[3]
    return (1/(1+c3))*(-(c1+2*c2*phi)**2 /np.cosh(c1*phi+c2*phi**2)**3+c3 *(-c4**2 /np.cosh(c4*phi)**3+c4**2 /np.cosh(c4*phi) *np.tanh(c4*phi)**2)-2 *c2 /np.cosh(c1*phi+c2*phi**2) *np.tanh(c1*phi+c2*phi**2)+(c1+2*c2*phi)**2 /np.cosh(c1*phi+c2*phi**2) *np.tanh(c1*phi+c2*phi**2)**2)
def dddf_sech(phi, *args):
    c1, c2, c3, c4 = args[0], args[1], args[2], args[3]
    return (1/(1+c3))*(-6* c2* (c1+2*c2*phi) /np.cosh(c1*phi+c2*phi**2)**3+c3* (5* c4**3 /np.cosh(c4*phi)**3 *np.tanh(c4*phi)-c4**3 /np.cosh(c4*phi) *np.tanh(c4*phi)**3)+5* (c1+2*c2*phi)**3 /np.cosh(c1*phi+c2*phi**2)**3 *np.tanh(c1*phi+c2*phi**2)+6* c2* (c1+2*c2*phi) /np.cosh(c1*phi+c2*phi**2) *np.tanh(c1*phi+c2*phi**2)**2-(c1+2*c2*phi)**3 /np.cosh(c1*phi+c2*phi**2) *np.tanh(c1*phi+c2*phi**2)**3)
def ddddf_sech(phi, *args):
    c1, c2, c3, c4 = args[0], args[1], args[2], args[3]
    return (1/(1+c3))*(-12* c2**2 /np.cosh(c1*phi+c2*phi**2)**3+5 *(c1+2*c2*phi)**4 /np.cosh(c1*phi+c2*phi**2)**5+c3 *(5 *c4**4 /np.cosh(c4*phi)**5-18* c4**4 /np.cosh(c4*phi)**3 *np.tanh(c4*phi)**2+c4**4 /np.cosh(c4*phi) *np.tanh(c4*phi)**4)+60* c2* (c1+2*c2*phi)**2 /np.cosh(c1*phi+c2*phi**2)**3 *np.tanh(c1*phi+c2*phi**2)+12* c2**2 /np.cosh(c1*phi+c2*phi**2) *np.tanh(c1*phi+c2*phi**2)**2-18 *(c1+2*c2*phi)**4 /np.cosh(c1*phi+c2*phi**2)**3 *np.tanh(c1*phi+c2*phi**2)**2-12 *c2* (c1+2*c2*phi)**2 /np.cosh(c1*phi+c2*phi**2) *np.tanh(c1*phi+c2*phi**2)**3+(c1+2*c2*phi)**4 /np.cosh(c1*phi+c2*phi**2) *np.tanh(c1*phi+c2*phi**2)**4)
def dddddf_sech(phi, *args):
    c1, c2, c3, c4 = args[0], args[1], args[2], args[3]
    return (1/(1+c3))*(100 *c2* (c1+2*c2*phi)**3 /np.cosh(c1*phi+c2*phi**2)**5+c3 *(-61* c4**5 /np.cosh(c4*phi)**5 *np.tanh(c4*phi)+58* c4**5 /np.cosh(c4*phi)**3 *np.tanh(c4*phi)**3-c4**5 /np.cosh(c4*phi) *np.tanh(c4*phi)**5)+300* c2**2 *(c1+2*c2*phi) /np.cosh(c1*phi+c2*phi**2)**3 *np.tanh(c1*phi+c2*phi**2)-61* (c1+2*c2*phi)**5 /np.cosh(c1*phi+c2*phi**2)**5 *np.tanh(c1*phi+c2*phi**2)-360* c2* (c1+2*c2*phi)**3 /np.cosh(c1*phi+c2*phi**2)**3 *np.tanh(c1*phi+c2*phi**2)**2-60* c2**2 *(c1+2*c2*phi) /np.cosh(c1*phi+c2*phi**2) *np.tanh(c1*phi+c2*phi**2)**3+58* (c1+2*c2*phi)**5 /np.cosh(c1*phi+c2*phi**2)**3 *np.tanh(c1*phi+c2*phi**2)**3+20* c2* (c1+2*c2*phi)**3 /np.cosh(c1*phi+c2*phi**2) *np.tanh(c1*phi+c2*phi**2)**4-(c1+2*c2*phi)**5 /np.cosh(c1*phi+c2*phi**2) *np.tanh(c1*phi+c2*phi**2)**5)

def dlogf_sech(phi, *args):
    return df_sech(phi, *args)/f_sech(phi, *args)


## f_tanh:
def f_tanh(t, *args):
    return args[0] + args[1]*np.tanh(args[2]*(t - args[3]))
def df_tanh(t, *args):
    return args[1]*args[2]*1.0/np.cosh(args[2]*(t - args[3]))**2.0
def ddf_tanh(t, *args):
    return - 2.0*args[1]*args[2]**2.0*np.sinh(args[2]*(t - args[3]))/np.cosh(args[2]*(t - args[3]))**3.0
def dddf_tanh(t, *args):
    return 2.0*args[1]*args[2]**3.0*(2.0*np.sinh(args[2]*(t - args[3]))**2.0 - 1.0)/np.cosh(args[2]*(t - args[3]))**4.0
def ddddf_tanh(t, *args):
    return - 8.0*args[1]*args[2]**4.0*np.sinh(args[2]*(t - args[3]))*(np.sinh(args[2]*(t - args[3]))**2.0 - 2.0)/np.cosh(args[2]*(t - args[3]))**5.0
def dddddf_tanh(t, *args):
    return 8.0*args[1]*args[2]**5.0*(2.0*np.sinh(args[2]*(t - args[3]))**4.0 - 11.0*np.sinh(args[2]*(t - args[3]))**2.0 + 2.0)/np.cosh(args[2]*(t - args[3]))**6.0

def dlogf_tanh(t, *args):
    return df_tanh(t, *args)/f_tanh(t, *args)


## f_tanh2:
def f_add(phi, *args):
    b4,b5,b6 = args[4],args[5],args[6]
    return  b4*(1.0+np.tanh(b6*(phi - b5)))*phi
def df_add(phi, *args):
    b4,b5,b6 = args[4],args[5],args[6]
    return b4*b6*phi/np.cosh(b6*(-b5+phi))**2.0 + b4*(1.0+np.tanh(b6*(-b5+phi)))
def ddf_add(phi, *args):
    b4,b5,b6 = args[4],args[5],args[6]
    return 2.0* b4*b6 /np.cosh(b6*(-b5+phi))**2 - 2.0*b4*b6**2.0*phi /np.cosh(b6*(-b5+phi))**2.0 *np.tanh(b6*(-b5+phi))
def dddf_add(phi, *args):
    b4,b5,b6 = args[4],args[5],args[6]
    return -6.0*b4* b6**2 /np.cosh(b6*(-b5+phi))**2 *np.tanh(b6*(-b5+phi))+b4* phi* (-2.0* b6**3.0 /np.cosh(b6*(-b5+phi))**4.0 + 4.0*b6**3.0 /np.cosh(b6*(-b5+phi))**2.0 *np.tanh(b6*(-b5+phi))**2)
def ddddf_add(phi, *args):
    b4,b5,b6 = args[4],args[5],args[6]
    return 4.0* b4* (-2.0* b6**3.0 /np.cosh(b6*(-b5+phi))**4.0 + 4.0* b6**3 /np.cosh(b6*(-b5+phi))**2 *np.tanh(b6*(-b5+phi))**2) + b4* phi* (16.0* b6**4.0 /np.cosh(b6*(-b5+phi))**4.0 *np.tanh(b6*(-b5+phi)) -8.0*b6**4.0 /np.cosh(b6* (-b5+phi))**2.0 *np.tanh(b6*(-b5+phi))**3.0)
def dddddf_add(phi, *args):
    b4,b5,b6 = args[4],args[5],args[6]
    return 5.0* b4* (16.0* b6**4.0 /np.cosh(b6*(-b5+phi))**4 *np.tanh(b6*(-b5+phi)) - 8.0* b6**4 /np.cosh(b6*(-b5+phi))**2 *np.tanh(b6*(-b5+phi))**3)+b4* phi* (16.0* b6**5 /np.cosh(b6*(-b5+phi))**6.0 - 88.0* b6**5 /np.cosh(b6*(-b5+phi))**4 *np.tanh(b6*(-b5+phi))**2 + 16.0* b6**5 /np.cosh(b6*(-b5+phi))**2 *np.tanh(b6*(-b5+phi))**4)

def f_tanh2(t, *args):
    return args[0] + args[1]*np.tanh(args[2]*(t - args[3])) + f_add(t, *args) + args[7]*np.exp(-args[8]*t)
def df_tanh2(t, *args):
    return args[1]*args[2]*1.0/np.cosh(args[2]*(t - args[3]))**2.0 + df_add(t, *args) - args[7]*args[8]*np.exp(-args[8]*t)
def ddf_tanh2(t, *args):
    return - 2.0*args[1]*args[2]**2.0*np.sinh(args[2]*(t - args[3]))/np.cosh(args[2]*(t - args[3]))**3.0 + ddf_add(t, *args) + args[7]*args[8]**2*np.exp(-args[8]*t)
def dddf_tanh2(t, *args):
    return 2.0*args[1]*args[2]**3.0*(2.0*np.sinh(args[2]*(t - args[3]))**2.0 - 1.0)/np.cosh(args[2]*(t - args[3]))**4.0 + dddf_add(t, *args) - args[7]*args[8]**3*np.exp(-args[8]*t)
def ddddf_tanh2(t, *args):
    return - 8.0*args[1]*args[2]**4.0*np.sinh(args[2]*(t - args[3]))*(np.sinh(args[2]*(t - args[3]))**2.0 - 2.0)/np.cosh(args[2]*(t - args[3]))**5.0 + ddddf_add(t, *args) + args[7]*args[8]**4*np.exp(-args[8]*t)
def dddddf_tanh2(t, *args):
    return 8.0*args[1]*args[2]**5.0*(2.0*np.sinh(args[2]*(t - args[3]))**4.0 - 11.0*np.sinh(args[2]*(t - args[3]))**2.0 + 2.0)/np.cosh(args[2]*(t - args[3]))**6.0 + dddddf_add(t, *args) - args[7]*args[8]**5*np.exp(-args[8]*t)

def dlogf_tanh2(t, *args):
    return df_tanh2(t, *args)/f_tanh2(t, *args)


## f_tanh3:
def f_tanh3(phi, *args):
    b0,b1,b2,b3,b4,b5 = args[0],args[1],args[2],args[3],args[4],args[5]
    return (1.0/2.0)* (b0+b1 *np.tanh(b2*(-b3+phi)))* (1.0-np.tanh(b4*(-b5+phi))) + args[6]*np.exp(-args[7]*phi)
def df_tanh3(phi, *args):
    b0,b1,b2,b3,b4,b5 = args[0],args[1],args[2],args[3],args[4],args[5]
    return -(1.0/2.0)* b4 /np.cosh(b4*(-b5+phi))**2 *(b0+b1 *np.tanh(b2*(-b3+phi))) + 1./2* b1*b2 /np.cosh(b2*(-b3+phi))**2 *(1.0-np.tanh(b4*(-b5+phi))) - args[6]*args[7]*np.exp(-args[7]*phi)
def ddf_tanh3(phi, *args):
    b0,b1,b2,b3,b4,b5 = args[0],args[1],args[2],args[3],args[4],args[5]
    return -b1* b2* b4 /np.cosh(b2*(-b3+phi))**2 /np.cosh(b4*(-b5+phi))**2-b1*b2**2 /np.cosh(b2*(-b3+phi))**2 *np.tanh(b2*(-b3+phi)) *(1.0-np.tanh(b4*(-b5+phi)))+b4**2 /np.cosh(b4*(-b5+phi))**2* (b0+b1 *np.tanh(b2*(-b3+phi))) *np.tanh(b4*(-b5+phi)) + args[6]*args[7]**2*np.exp(-args[7]*phi)
def dddf_tanh3(phi, *args):
    b0,b1,b2,b3,b4,b5 = args[0],args[1],args[2],args[3],args[4],args[5]
    return 3.0* b1*b2**2* b4 /np.cosh(b2*(-b3+phi))**2 /np.cosh(b4*(-b5+phi))**2 *np.tanh(b2*(-b3+phi))+1./2* b1* (-2.0* b2**3 /np.cosh(b2*(-b3+phi))**4 + 4.0* b2**3 /np.cosh(b2*(-b3+phi))**2 *np.tanh(b2*(-b3+phi))**2)* (1.0-np.tanh(b4*(-b5+phi)))+3.0* b1* b2* b4**2 /np.cosh(b2*(-b3+phi))**2 /np.cosh(b4*(-b5+phi))**2 *np.tanh(b4*(-b5+phi))+1.0/2* (b0+b1 *np.tanh(b2*(-b3+phi)))* (2.0* b4**3 /np.cosh(b4*(-b5+phi))**4 - 4.0* b4**3 /np.cosh(b4*(-b5+phi))**2 *np.tanh(b4*(-b5+phi))**2) - args[6]*args[7]**3*np.exp(-args[7]*phi)
def ddddf_tanh3(phi, *args):
    b0,b1,b2,b3,b4,b5 = args[0],args[1],args[2],args[3],args[4],args[5]
    return ( -2.0* b1*b4 /np.cosh(b4*(-b5+phi))**2 *(-2.0* b2**3 /np.cosh(b2*(-b3+phi))**4 + 4.0* b2**3 /np.cosh(b2*(-b3+phi))**2 *np.tanh(b2*(-b3+phi))**2) +
             1.0/2* b1* (16.0* b2**4 /np.cosh(b2*(-b3+phi))**4 *np.tanh(b2*(-b3+phi)) - 8.0* b2**4 /np.cosh(b2*(-b3+phi))**2 *np.tanh(b2*(-b3+phi))**3)* (1.0-np.tanh(b4*(-b5+phi))) -
             12.0* b1*b2**2* b4**2 /np.cosh(b2*(-b3+phi))**2 /np.cosh(b4*(-b5+phi))**2 *np.tanh(b2*(-b3+phi)) *np.tanh(b4*(-b5+phi)) + 2.0* b1*b2 /np.cosh(b2*(-b3+phi))**2 *
             (2.0* b4**3 /np.cosh(b4*(-b5+phi))**4 - 4.0* b4**3 /np.cosh(b4*(-b5+phi))**2 *np.tanh(b4*(-b5+phi))**2) + 1.0/2* (b0+b1 *np.tanh(b2*(-b3+phi)))*
             (-16.0* b4**4 /np.cosh(b4*(-b5+phi))**4 *np.tanh(b4*(-b5+phi)) + 8.0* b4**4 /np.cosh(b4*(-b5+phi))**2 *np.tanh(b4*(-b5+phi))**3)
              + args[6]*args[7]**4*np.exp(-args[7]*phi) )
def dddddf_tanh3(phi, *args):
    b0,b1,b2,b3,b4,b5 = args[0],args[1],args[2],args[3],args[4],args[5]
    return ( -(5.0/2.0)* b1*b4 /np.cosh(b4*(-b5+phi))**2 *(16.0* b2**4 /np.cosh(b2*(-b3+phi))**4 *np.tanh(b2*(-b3+phi)) - 8.0* b2**4 /np.cosh(b2*(-b3+phi))**2 *np.tanh(b2*(-b3+phi))**3) +
             1.0/2* b1* (16.0* b2**5 /np.cosh(b2*(-b3+phi))**6 - 88.0* b2**5 /np.cosh(b2*(-b3+phi))**4 *np.tanh(b2*(-b3+phi))**2 + 16.0* b2**5 /np.cosh(b2*(-b3+phi))**2 *
             np.tanh(b2*(-b3+phi))**4)* (1.0-np.tanh(b4*(-b5+phi))) + 10.0* b1*b4**2 /np.cosh(b4*(-b5+phi))**2 *(-2.0* b2**3 /np.cosh(b2*(-b3+phi))**4 + 4.0* b2**3 /
             np.cosh(b2*(-b3+phi))**2 *np.tanh(b2*(-b3+phi))**2) *np.tanh(b4*(-b5+phi)) - 10.0* b1*b2**2 /np.cosh(b2*(-b3+phi))**2 *np.tanh(b2*(-b3+phi))* (2.0* b4**3 /
             np.cosh(b4*(-b5+phi))**4 - 4.0* b4**3 /np.cosh(b4*(-b5+phi))**2 *np.tanh(b4*(-b5+phi))**2) + 5.0/2* b1*b2 /np.cosh(b2*(-b3+phi))**2 *(-16.0* b4**4 /np.cosh(b4*(-b5+phi))**4 *
             np.tanh(b4*(-b5+phi)) + 8.0* b4**4 /np.cosh(b4*(-b5+phi))**2 *np.tanh(b4*(-b5+phi))**3) + 1.0/2* (b0+b1 *np.tanh(b2*(-b3+phi)))* (-16.0* b4**5 /np.cosh(b4*(-b5+phi))**6 +
             88.0* b4**5 /np.cosh(b4*(-b5+phi))**4 *np.tanh(b4*(-b5+phi))**2 - 16.0* b4**5 /np.cosh(b4*(-b5+phi))**2 *np.tanh(b4*(-b5+phi))**4)
             + args[6]*args[7]**5*np.exp(-args[7]*phi) )

def dlogf_tanh3(t, *args):
    return df_tanh3(t, *args)/f_tanh3(t, *args)


## f_tanhexp:
def f_tanhexp(t, *args):
    return args[0] + args[1]*np.tanh(args[2]*(t - args[3])) + args[4]*np.exp(-args[5]*t)
def df_tanhexp(t, *args):
    return args[1]*args[2]*1.0/np.cosh(args[2]*(t - args[3]))**2.0 - args[4]*args[5]*np.exp(-args[5]*t)
def ddf_tanhexp(t, *args):
    return - 2.0*args[1]*args[2]**2.0*np.sinh(args[2]*(t - args[3]))/np.cosh(args[2]*(t - args[3]))**3.0 + args[4]*args[5]**2*np.exp(-args[5]*t)
def dddf_tanhexp(t, *args):
    return 2.0*args[1]*args[2]**3.0*(2.0*np.sinh(args[2]*(t - args[3]))**2.0 - 1.0)/np.cosh(args[2]*(t - args[3]))**4.0 - args[4]*args[5]**3*np.exp(-args[5]*t)
def ddddf_tanhexp(t, *args):
    return - 8.0*args[1]*args[2]**4.0*np.sinh(args[2]*(t - args[3]))*(np.sinh(args[2]*(t - args[3]))**2.0 - 2.0)/np.cosh(args[2]*(t - args[3]))**5.0 + args[4]*args[5]**4*np.exp(-args[5]*t)
def dddddf_tanhexp(t, *args):
    return 8.0*args[1]*args[2]**5.0*(2.0*np.sinh(args[2]*(t - args[3]))**4.0 - 11.0*np.sinh(args[2]*(t - args[3]))**2.0 + 2.0)/np.cosh(args[2]*(t - args[3]))**6.0 - args[4]*args[5]**5*np.exp(-args[5]*t)

def dlogf_tanhexp(t, *args):
    return df_tanhexp(t, *args)/f_tanhexp(t, *args)


## others:
def f_poly4(t, *args):
    # print args
    # if args[3] == 0:
    #     print buh
    return args[0] + args[1]*t + args[2]*t**2.0 + args[3]*t**3.0 + args[4]*t**4.0
def df_poly4(t, *args):
    return args[1] + 2.0*args[2]*t + 3.0*args[3]*t**2.0 + 4.0*args[4]*t**3.0
def ddf_poly4(t, *args):
    return 2.0*args[2] + 6.0*args[3]*t + 12.0*args[4]*t**2.0
def dddf_poly4(t, *args):
    return 6.0*args[3] + 24.0*args[4]*t
def ddddf_poly4(t, *args):
    return 24.0*args[4]
def dddddf_poly4(t, *args):
    return 0

def dlogf_poly4(t, *args):
    return df_poly4(t, *args)/f_poly4(t, *args)

def f_I_poly1(t, *args):
    return f_I(t, *args[:3]) + f_poly4(t, *(args[3:5]+(0, 0, 0)))
def df_I_poly1(t, *args):
    return df_I(t, *args[:3]) + df_poly4(t, *(args[3:5]+(0, 0, 0)))
def ddf_I_poly1(t, *args):
    return f_I(t, *args[:3]) + ddf_poly4(t, *(args[3:5]+(0, 0, 0)))
def dddf_I_poly1(t, *args):
    return dddf_I(t, *args[:3]) + dddf_poly4(t, *(args[3:5]+(0, 0, 0)))
def ddddf_I_poly1(t, *args):
    return ddddf_I(t, *args[:3]) + ddddf_poly4(t, *(args[3:5]+(0, 0, 0)))
def dddddf_I_poly1(t, *args):
    return dddddf_I(t, *args[:3]) + dddddf_poly4(t, *(args[3:5]+(0, 0, 0)))

def dlogf_I_poly1(t, *args):
    return df_I_poly1(t, *args)/f_I_poly1(t, *args)

def f_I_poly2(t, *args):
    return f_I(t, *args[:3]) + f_poly4(t, *(args[3:6]+(0, 0)))
def df_I_poly2(t, *args):
    return df_I(t, *args[:3]) + df_poly4(t, *(args[3:6]+(0, 0)))
def ddf_I_poly2(t, *args):
    return f_I(t, *args[:3]) + ddf_poly4(t, *(args[3:6]+(0, 0)))
def dddf_I_poly2(t, *args):
    return dddf_I(t, *args[:3]) + dddf_poly4(t, *(args[3:6]+(0, 0)))
def ddddf_I_poly2(t, *args):
    return ddddf_I(t, *args[:3]) + ddddf_poly4(t, *(args[3:6]+(0, 0)))
def dddddf_I_poly2(t, *args):
    return dddddf_I(t, *args[:3]) + dddddf_poly4(t, *(args[3:6]+(0, 0)))

def dlogf_I_poly2(t, *args):
    return df_I_poly2(t, *args)/f_I_poly2(t, *args)

def f_coshq(t, *args):
    return args[0]/np.cosh(args[1]*(t - args[2]))**2.0

def f_cq_poly1(t, *args):
    return f_coshq(t, *args[:3]) + f_poly4(t, *(args[3:5]+(0, 0, 0)))

def f_cq_poly2(t, *args):
    return f_coshq(t, *args[:3]) + f_poly4(t, *(args[3:6]+(0, 0)))

def f_cq_th(t, *args):
    return f_coshq(t, *args[:3]) + f_tanh(t, *args[3:])

def f_cq_thsa(t, *args):
    return f_coshq(t, *args[:3]) + f_tanh(t, *(args[3:5] + args[1:3]))


##=======================================================================================================================================================
fs =      {'f_I':f_I, 'f_no':f_no, 'f_fi':f_fi, 'f_tanh':f_tanh, 'f_sech':f_sech, 'f_tanh2':f_tanh2, 'f_tanh3':f_tanh3, 'f_tanhexp':f_tanhexp, 'f_poly4':f_poly4, 'f_I_poly1':f_I_poly1, 'f_I_poly2':f_I_poly2,
           'f_coshq':f_coshq, 'f_cq_poly1':f_cq_poly1, 'f_cq_poly2':f_cq_poly2, 'f_cq_th':f_cq_th, 'f_cq_thsa':f_cq_thsa}
dfs =     {'f_I':df_I, 'f_no':df_no, 'f_fi':df_fi, 'f_tanh':df_tanh, 'f_sech':df_sech, 'f_tanh2':df_tanh2, 'f_tanh3':df_tanh3, 'f_tanhexp':df_tanhexp, 'f_poly4':df_poly4, 'f_I_poly1':df_I_poly1, 'f_I_poly2':df_I_poly2}
ddfs =    {'f_I':ddf_I, 'f_no':ddf_no, 'f_fi':ddf_fi, 'f_tanh':ddf_tanh, 'f_sech':ddf_sech, 'f_tanh2':ddf_tanh2, 'f_tanh3':ddf_tanh3, 'f_tanhexp':ddf_tanhexp, 'f_poly4':ddf_poly4, 'f_I_poly1':ddf_I_poly1, 'f_I_poly2':ddf_I_poly2}
dddfs =   {'f_I':dddf_I, 'f_no':dddf_no, 'f_fi':dddf_fi, 'f_tanh':dddf_tanh, 'f_sech':dddf_sech, 'f_tanh2':dddf_tanh2, 'f_tanh3':dddf_tanh3, 'f_tanhexp':dddf_tanhexp, 'f_poly4':dddf_poly4, 'f_I_poly1':dddf_I_poly1, 'f_I_poly2':dddf_I_poly2}
ddddfs =  {'f_I':ddddf_I, 'f_no':ddddf_no, 'f_fi':ddddf_fi, 'f_tanh':ddddf_tanh, 'f_sech':ddddf_sech, 'f_tanh2':ddddf_tanh2, 'f_tanh3':ddddf_tanh3, 'f_tanhexp':ddddf_tanhexp, 'f_poly4':ddddf_poly4, 'f_I_poly1':ddddf_I_poly1, 'f_I_poly2':ddddf_I_poly2}
dddddfs = {'f_I':dddddf_I, 'f_no':dddddf_no, 'f_fi':dddddf_fi, 'f_tanh':dddddf_tanh, 'f_sech':dddddf_sech, 'f_tanh2':dddddf_tanh2, 'f_tanh3':dddddf_tanh3, 'f_tanhexp':dddddf_tanhexp, 'f_poly4':dddddf_poly4, 'f_I_poly1':dddddf_I_poly1, 'f_I_poly2':dddddf_I_poly2}
dlogfs =  {'f_I':dlogf_I, 'f_no':dlogf_no, 'f_fi':dlogf_fi, 'f_tanh':dlogf_tanh, 'f_sech':dlogf_sech, 'f_tanh2':dlogf_tanh2, 'f_tanh3':dlogf_tanh3, 'f_tanhexp':dlogf_tanhexp, 'f_poly4':dlogf_poly4, 'f_I_poly1':dlogf_I_poly1, 'f_I_poly2':dlogf_I_poly2}
