import numpy as np

def TD_scale(TD_grid, lambdas):
    TD_grid_scaled = np.zeros_like(TD_grid)
    for k in range(0, 4):
        TD_grid_scaled[:, :, k] = lambdas[k]*TD_grid[:, :, k]

    return [TD_grid_scaled, TD_grid, lambdas]

#def TD_scale(TD_grid, lambdas):
#    TD_grid_scaled = np.zeros_like(TD_grid)
#    for k in range(0, 4):
#        TD_grid_scaled[:, :, k] = lambdas[k]*TD_grid[:, :, k]
#
#    return [TD_grid_scaled, TD_grid, lambdas]

def TD_scale_isen(TD_isen, lambdas):
    TD_isen_scaled = np.zeros_like(TD_isen)
    for k in range(0, 4):
        TD_isen_scaled[:, k] = lambdas[k]*TD_isen[:, k]

    return [TD_isen_scaled, TD_isen, lambdas]

def NNmatrix(i0, j0, i_incr, j_incr, TD_grid, TD_inds):
    if i_incr == 1 and j_incr == 1:
        i = i0
        j = j0
    elif i_incr == -1 and j_incr == 1:
        i = i0 - 1
        j = j0
    elif i_incr == 1 and j_incr == -1:
        i = i0
        j = j0 - 1
    elif i_incr == -1 and j_incr == -1:
        i = i0 - 1
        j = j0 - 1
    up = np.array([TD_grid[i + 1, j, TD_inds[0]] - TD_grid[i, j, TD_inds[0]], TD_grid[i, j + 1, TD_inds[0]] - TD_grid[i, j, TD_inds[0]]])
    down = np.array([TD_grid[i + 1, j, TD_inds[1]] - TD_grid[i, j, TD_inds[1]], TD_grid[i, j + 1, TD_inds[1]] - TD_grid[i, j, TD_inds[1]]])
    return np.vstack((up, down))

def J_calc_fd(TD_grid, phi0_pts, Phi1_pts, phiPhi_grid):
    J = np.zeros((phi0_pts, Phi1_pts))
    j_maxs = np.zeros(phi0_pts - 1, dtype=int)
    j_max = Phi1_pts - 1
    for i in range(0, phi0_pts - 1):
        for j in range(0, Phi1_pts - 1):
            S_Tmu = NNmatrix(i, j, 1, 1, TD_grid, [0, 2]) #T and mu
            S_srho = NNmatrix(i, j, 1, 1, TD_grid, [1, 3]) #s and rho
            J_Tmu = np.linalg.det(S_Tmu)
            J_srho = np.linalg.det(S_srho)
            J[i, j] = J_srho/J_Tmu
            (sgn, logdet) =  np.linalg.slogdet(S_Tmu)
            J_Tmu2 = sgn*np.exp(logdet)
            (sgn, logdet) =  np.linalg.slogdet(S_srho)
            J_srho2 = sgn*np.exp(logdet)
            J2 = J_srho2/J_Tmu2
#            print i, j
#            print phiPhi_grid[i, j], phiPhi_grid[i + 1, j], phiPhi_grid[i, j + 1], phiPhi_grid[i + 1, j + 1]
#            print TD_grid[i:i+2, j:j+2, 0], '\n', TD_grid[i:i+2, j:j+2, 2]
#            print 'S_Tmu:'
#            print S_Tmu
#            print 'S_srho:'
#            print S_srho
#            print 'Js:', J_Tmu, J_srho, J[i, j], J2
            if j + 2 <= j_max:
                if TD_grid[i, j + 2, 0] == 0:
                    j_maxs[i] = j + 1
                    break
            else:
                j_maxs[i] = j_max

    print j_maxs
    #top slice of J
    i = phi0_pts - 1 #i_max
    for j in range(0, Phi1_pts - 1): # -1 to exclue upper right Phi_1
        S_Tmu = NNmatrix(i, j, -1, 1, TD_grid, [0, 2])
        S_srho = NNmatrix(i, j, -1, 1, TD_grid, [1, 3])
        J_Tmu = np.linalg.det(S_Tmu)
        J_srho = np.linalg.det(S_srho)
        J[i, j] = J_srho/J_Tmu

    #rightmost slice of J
    i = 0
    for j in j_maxs: # len(j_maxs) excludes last phi_0
        S_Tmu = NNmatrix(i, j, 1, -1, TD_grid, [0, 2])
        S_srho = NNmatrix(i, j, 1, -1, TD_grid, [1, 3])
        J_Tmu = np.linalg.det(S_Tmu)
        J_srho = np.linalg.det(S_srho)
        J[i, j] = J_srho/J_Tmu
        print 'j_max =', j, 'i =', i
        i += 1

    #upper right corner of J
    i = phi0_pts - 1
    j = Phi1_pts - 1
    S_Tmu = NNmatrix(i, j, -1, -1, TD_grid, [0, 2])
    S_srho = NNmatrix(i, j, -1, -1, TD_grid, [1, 3])
    J_Tmu = np.linalg.det(S_Tmu)
    J_srho = np.linalg.det(S_srho)
    J[i, j] = J_srho/J_Tmu

    return J
