import numpy as np

N_c = 3.0
N_f = 3.0
K = np.pi**2.0/90.0
d_q = 2.0*2.0*N_f*N_c ## quark/antiquark*2 spin projections*N_flavors*N_colors
## with mu_B and d_q = 36: chi_2 = d_q/12/9 = 1/3
## => chi_2/T^2 = 2*d_q*K*c2 = 1/3

def p_SB_G(T):
    return (N_c**2.0 - 1.0)*np.pi**2.0/45.0*T**4.0

def s_SB_G(T):
    return 4.0*(N_c**2.0 - 1.0)*np.pi**2.0/45.0*T**3.0

mu_type = 'mu_B'
c1 = 7.0/8.0
if mu_type == 'mu_B':
    c2 = 15.0/(4.0*np.pi**2.0)
    c3 = 15.0/(8.0*np.pi**4.0)
elif mu_type == 'mu_f':
    c2 = 5.0/(12.0*np.pi**2.0)
    c3 = 5.0/(216.0*np.pi**4.0)

def p_SB_F(T, mu):
    if mu_type == 'mu_B':
        mu = mu/3.0
    return d_q*K*(c1*T**4.0 + c2*mu**2.0*T**2.0 + c3*mu**4.0)
    
def s_SB_F(T, mu): #dp/dT|_mu
    if mu_type == 'mu_B':
        mu = mu/3.0
    return d_q*K*(4.0*c1*T**3.0 + 2.0*c2*mu**2.0*T)
    
def rho_SB_F(T, mu): #dp/dmu|_T
    if mu_type == 'mu_B':
        mu = mu/3.0
    return d_q*K*(2.0*c2*mu*T**2.0 + 4.0*c3*mu**3.0)

def Cmu_SB_F(T, mu): #T*(d^2p/dT^2)
    if mu_type == 'mu_B':
        mu = mu/3.0
    return d_q*K*T*(12.0*c1*T**2.0 + 2.0*c2*mu**2.0)
    
def chi_SB_F(T, mu): #d^2p/dmu^2
    if mu_type == 'mu_B':
        mu = mu/3.0
    return d_q*K*(2.0*c2*T**2.0 + 12.0*c3*mu**2.0)

def dsdmu_SB_F(T, mu):
    if mu_type == 'mu_B':
        mu = mu/3.0
    return 4.0*d_q*K*N_c*c2*mu*T
    
def drhodT_SB_F(T, mu):
    if mu_type == 'mu_B':
        mu = mu/3.0
    return 4.0*d_q*K*N_c*c2*mu*T
    
def hessian_p_SB_F(T, mu):
    return np.array([[Cmu_SB_F(T, mu)/T, dsdmu_SB_F(T, mu)], [drhodT_SB_F(T, mu), chi_SB_F(T, mu)]])
    

def s_SB(T, mu):
    return s_SB_G(T) + s_SB_F(T, mu)

def rho_SB(T, mu):
    return rho_SB_F(T, mu)


