"""
physics.py
Written by Sindre Stenen Blakseth, 2021.
Numerical simulation of the 1D heat equation and generation of corrective source terms.
"""

########################################################################################################################
# Package imports.
import sys

import numpy as np
import scipy.linalg
from scipy.sparse import diags

########################################################################################################################
# File imports.

import config
import util

########################################################################################################################

def get_A_matrix(cfg):
    N_x = cfg.N_x
    N_y = cfg.N_y
    N = N_x * N_y

    dx = cfg.dx
    dy = cfg.dy
    dt = cfg.dt
    alpha = 1 # Hard-coded, assumes c_V, rho and k are all unity.
    r_x = alpha * dt / (dx**2)
    r_y = alpha * dt / (dy**2)

    main_diag = np.ones(N) * (1 + 2*r_x + 2*r_y)
    for i in range(N):
        if i < N_x or i >= N - N_x:
            main_diag[i] += r_y
        if i % N_x == 0 or i % N_x == N_x - 1:
            main_diag[i] += r_x

    sub_diag = -np.ones(N-1) * r_x
    sup_diag = -np.ones(N-1) * r_x
    for i in range(N-2):
        if int(i % N_x) == int(N_x - 1):
            sub_diag[i] = 0.0
            sup_diag[i] = 0.0

    low_diag = -np.ones(N - N_x) * r_y
    high_diag = -np.ones(N - N_x) * r_y

    diagonals = [low_diag, sub_diag, main_diag, sup_diag, high_diag]
    offsets   = [-N_x, -1, 0, 1, N_x]

    A = diags(diagonals, offsets).toarray()

    np.set_printoptions(suppress=True, linewidth=np.nan, threshold=sys.maxsize)

    return A

def get_b_vector(cfg, T_old, t_old, alpha, get_src):
    N_x = cfg.N_x
    dx = cfg.dx
    dy = cfg.dy
    dt = cfg.dt
    a = 1  # Hard-coded, assumes c_V, rho and k are all unity.
    r_x = a * dt / (dx ** 2)
    r_y = a * dt / (dy ** 2)
    t_new = t_old + dt

    b = T_old[1:-1,1:-1].flatten('C')
    for i, y in enumerate(cfg.y_nodes[1:-1]):
        for j, x in enumerate(cfg.x_nodes[1:-1]):
            if x == cfg.x_nodes[1]:
                #print("y:", y)
                #print("T_a:",cfg.get_T_a(y, t_new, alpha))
                #print("1 boundary:", 2 * r_x * cfg.get_T_a(y, t_new, alpha))
                b[i*N_x + j] += 2 * r_x * cfg.get_T_a(y, t_new, alpha)
            elif x == cfg.x_nodes[-2]:
                b[i*N_x + j] += 2 * r_x * cfg.get_T_b(y, t_new, alpha)
            if y == cfg.y_nodes[1]:
                b[i*N_x + j] += 2 * r_y * cfg.get_T_c(x, t_new, alpha)
            elif y == cfg.y_nodes[-2]:
                b[i*N_x + j] += 2 * r_y * cfg.get_T_d(x, t_new, alpha)
    b += dt * get_src(cfg.x_nodes[1:-1], cfg.y_nodes[1:-1], t_new, alpha).flatten(order='C')

    return b

########################################################################################################################

def simulate_2D(cfg, T_old, t_start, t_end, alpha, get_src, cor_src):
    A = get_A_matrix(cfg)
    time = t_start

    while time < t_end:
        b = get_b_vector(cfg, T_old, time, alpha, get_src)
        b += cor_src.flatten(order='C')
        T_interior_flat = scipy.linalg.solve(A, b)
        T_interior = T_interior_flat.reshape((cfg.N_x, cfg.N_y), order='C')

        t_new = np.around(time + cfg.dt, decimals=10)

        T_new = np.zeros((cfg.N_x + 2, cfg.N_y + 2))
        T_new[0,:]        = cfg.get_T_a(cfg.y_nodes, t_new, alpha)
        T_new[-1,:]       = cfg.get_T_b(cfg.y_nodes, t_new, alpha)
        T_new[:,0]        = cfg.get_T_c(cfg.x_nodes, t_new, alpha)
        T_new[:,-1]       = cfg.get_T_d(cfg.x_nodes, t_new, alpha)
        T_new[1:-1, 1:-1] = T_interior

        T_old = T_new
        time  = t_new

    return T_new

########################################################################################################################

def get_corrective_src_term_2D(cfg, T_old, T_new, t_old, alpha, get_src):
    A = get_A_matrix(cfg)
    b = get_b_vector(cfg, T_old, t_old, alpha, get_src)
    sigma_corr = np.dot(A, T_new[1:-1, 1:-1].flatten()) - b
    return sigma_corr.reshape((cfg.N_x, cfg.N_y))

########################################################################################################################

def main():
    N_js = [3, 9, 27, 81]#, 9, 27]#, 81, 81*3, 81*9]
    errors = []
    for N_j in N_js:
        cfg = config.Config(
            use_GPU=config.use_GPU,
            group_name=config.group_name,
            run_name=config.run_names[0][0],
            system=config.systems[0],
            data_tag=config.data_tags[0],
            model_key=config.model_keys[0],
            do_train=False,
            do_test=False,
            N_x=N_j,
            model_type=config.model_type,
        )
        t_start = 0.0
        alpha = 0.7
        T0 = np.zeros((cfg.N_x + 2, cfg.N_y + 2))
        for i, y in enumerate(cfg.y_nodes):
            for j, x in enumerate(cfg.x_nodes):
                T0[j][i] = cfg.get_T0(x, y, alpha)
        #print("T0:", T0)
        T = simulate_2D(cfg, T0, t_start, cfg.t_end, alpha, cfg.get_q_hat, np.zeros((cfg.N_x, cfg.N_y)))
        #print("T:", T)
        T_exact = np.zeros((cfg.N_x + 2, cfg.N_y + 2))
        for i, y in enumerate(cfg.y_nodes):
            for j, x in enumerate(cfg.x_nodes):
                T_exact[j][i] = cfg.get_T_exact(x, y, cfg.t_end, alpha)
        #print("T_exact:", T_exact)

        error = util.get_disc_L2_norm(T - T_exact)
        print("error:", error)
        print("scaling:", np.sqrt(cfg.dx * cfg.dy))
        scaled_error = error * np.sqrt(cfg.dx * cfg.dy)
        print("scaled_error:", scaled_error)
        errors.append(scaled_error)

        cor_src = get_corrective_src_term_2D(cfg, T0, T_exact, t_start, alpha, cfg.get_q_hat)
        T_cor = simulate_2D(cfg, T0, t_start, cfg.t_end, alpha, cfg.get_q_hat, cor_src)
        print("T_cor:  ", T_cor)
        print("T_exact:", T_exact)
        np.testing.assert_allclose(T_cor, T_exact, rtol=1e-10, atol=1e-10)

    print("errors:", errors)

    for i in range(len(errors) - 1):
        print("ratio", np.log(errors[i] / errors[i+1]) / np.log(3))


########################################################################################################################

if __name__ == "__main__":
    main()

########################################################################################################################