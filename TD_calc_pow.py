import numpy as np
from backbone_g_5_VRY import EFEs_solver_pw, get_UV_coeffs, get_UV_region, TD_from_UV_coeffs, V, f, Delta_from_Vargs, \
    r__tol
#global r_mid
kappa_5 = 1.0

def TD_calc_pointwise(phi_0, Phi_1r, V_args, f_args, r_mid):
    Phi_1_max = np.sqrt(-2.0*V(phi_0, *V_args)/f(phi_0, *f_args))
    Phi_1 = Phi_1r*Phi_1_max
    Q_G = f(phi_0, *f_args)*Phi_1
    f0 = f(0, *f_args)

    Delta = Delta_from_Vargs(V_args)
    a_tol = 1e-80 #10.0**( - (4.0*(4.0 - Delta)*r_inf) - 16)
    eps = 1e-6
    #print 'Delta =', Delta, 'a_tol =', a_tol
    #global r_mid

    print 'r_mid =', r_mid
    metric_sol = EFEs_solver_pw(phi_0, Phi_1, eps, r_mid, V_args, f_args, a_tol, r__tol, Delta)

    if metric_sol == 0: # in case there is no BH solution
        return 0

    solution_UV = get_UV_region(metric_sol, 'r_index', -50)
    UV_coeffs = get_UV_coeffs(solution_UV, f_args, Delta)
    TD_point = TD_from_UV_coeffs(UV_coeffs, Q_G, Delta, kappa_5, f0)
    #print TD_point

    if Phi_1/Phi_1_max < 0.60:
        if V_args[0] == 'V_I':
            r_mid = metric_sol['r'][-1]/40.0*(1.0 + 2.0*Phi_1/Phi_1_max)
        elif V_args[0] == 'V_VI':
            r_mid = metric_sol['r'][-1]/3.0*(1.0 + 2.0*Phi_1/Phi_1_max)
        elif V_args[0] == 'V_rpdb_pl_s':
            r_mid = metric_sol['r'][-1]/2.0
    else:
        if V_args[0] == 'V_I':
            r_mid = metric_sol['r'][-1]/30.0#*(1.0 + 20.0*Phi_1/Phi1_max)
        elif V_args[0] == 'V_VI':
            r_mid = metric_sol['r'][-1]/2.0
        elif V_args[0] == 'V_rpdb_pl_s':
            r_mid = metric_sol['r'][-1]/2.0*(1.0 + 10.0*Phi_1/Phi_1_max)
            if r_mid > 10.0:
                r_mid = 18.0
                print '=== changed high r_mid:'

    print 'r_mid =', r_mid

    return [TD_point, metric_sol, r_mid]
