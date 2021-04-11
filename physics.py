"""
physics.py
Written by Sindre Stenen Blakseth, 2021.
Numerical simulation of the 1D heat equation and generation of corrective source terms.
"""

########################################################################################################################
# Package imports.
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
from scipy.sparse import diags

########################################################################################################################
# File imports.

import config
import exact_solver
import util

########################################################################################################################

def get_init_V_mtx(cfg):
    V_mtx = np.zeros((3, cfg.x_nodes.shape[0]))
    for i in range(cfg.x_nodes.shape[0]):
        if cfg.x_nodes[i] <= cfg.x_split:
            V_mtx[0, i] = cfg.init_p1
            V_mtx[1, i] = cfg.init_u1
            V_mtx[2, i] = cfg.init_T1
        else:
            V_mtx[0, i] = cfg.init_p2
            V_mtx[1, i] = cfg.init_u2
            V_mtx[2, i] = cfg.init_T2
    return V_mtx

def LxF_flux(U_mtx, F_mtx, dt, dx):
    return 0.5 * ( F_mtx[:, :-1] + F_mtx[:, 1:] - (dx/dt)*(U_mtx[:, 1:] - U_mtx[:, 0:-1]) )

def find_c(cfg, T_vec):
    if np.amin(T_vec) < 0:
        raise Exception("Negative T in find_c.")
    return np.sqrt((cfg.gamma - 1)*(cfg.gamma*cfg.c_V)*T_vec)

def find_dt(u_vec, c_vec, CFL, dx):
    lam = np.amax(np.abs(u_vec) + c_vec)
    return CFL*dx/lam

def check_valid_dt(cfg, u_vec, c_vec, CFL, dx):
    lam = np.amax(np.abs(u_vec) + c_vec)
    if cfg.dt > CFL*dx/lam:
        print("Time step " + str(cfg.dt) + " is larger than the maximum allowed value " + str(CFL*dx/lam) + ".")
        raise Exception("Invalid dt.")

def interior_step(U_mtx, F_est, dt, dx, corr_src):
    return U_mtx[:, 1:-1] - (dt/dx)*(F_est[:, 1:] - F_est[:, :-1]) + corr_src

def find_implicit_vars(cfg, U_mtx):
    V_mtx = np.zeros_like(U_mtx)

    V_mtx[1,:] = U_mtx[1,:] / U_mtx[0,:]

    e = U_mtx[2,:] / U_mtx[0,:] - 0.5*(V_mtx[1,:]**2)

    if np.amin(e) < 0:
        raise Exception("Negative e in find_implicit_vars")

    V_mtx[2,:] = e/cfg.c_V
    V_mtx[0,:] = (cfg.gamma - 1)*U_mtx[0,:]*e

    return V_mtx

def find_flux(U_mtx, V_mtx):
    F_mtx = np.zeros_like(U_mtx)
    F_mtx[0,:] = U_mtx[1,:]
    F_mtx[1,:] = U_mtx[1,:] * V_mtx[1,:] + V_mtx[0,:]
    F_mtx[2,:] = (U_mtx[2,:] + V_mtx[0,:]) * V_mtx[1,:]
    return F_mtx

def setup_euler(cfg, V_mtx):

    U_mtx = np.zeros_like(V_mtx)

    e = cfg.c_V * V_mtx[2,:]
    U_mtx[0,:] = V_mtx[0,:] / ((cfg.gamma - 1)*e)
    U_mtx[1,:] = U_mtx[0,:] * V_mtx[1,:]
    U_mtx[2,:] = U_mtx[0,:]*e + 0.5*U_mtx[0,:]*V_mtx[1,:]**2

    F_mtx = find_flux(U_mtx, V_mtx)

    return U_mtx, F_mtx


def solve(cfg, V_mtx, U_mtx, F_mtx, T_vec, dx, tmax, CFL, corr_src):

    counter = 0
    t = 0.0
    dt = cfg.dt
    c_vec = find_c(cfg, T_vec)
    u_vec = V_mtx[1, :]

    while True:
        check_valid_dt(cfg, u_vec, c_vec, CFL, dx)
        t = np.around(t + cfg.dt, decimals=10)
        print("t =", t)

        F_est = LxF_flux(U_mtx, F_mtx, dt, dx)

        U_mtx[:, 1:-1] = interior_step(U_mtx, F_est, dt, dx, corr_src)

        U_mtx[:, 0] = U_mtx[:, 1]
        U_mtx[:, -1] = U_mtx[:, -2]

        V_mtx = find_implicit_vars(cfg, U_mtx)

        T_vec = V_mtx[2,:]

        c_vec = find_c(cfg, T_vec)
        u_vec = V_mtx[1, :]

        F_mtx = find_flux(U_mtx, V_mtx)

        if t >= tmax:
            break
        counter += 1

    return U_mtx, V_mtx



def simulate_euler(cfg, t_end):
    NJ = cfg.NJ # Includes boundary nodes.
    dx = cfg.dx
    V_mtx = np.zeros((3, NJ))
    T_vec = np.zeros(NJ)

    tmax = t_end
    CFL  = cfg.CFL

    for i in range(NJ):
        if cfg.x_nodes[i] <= cfg.x_split:
            V_mtx[0, i] = cfg.init_p1
            V_mtx[1, i] = cfg.init_u1
            V_mtx[2, i] = cfg.init_T1
        else:
            V_mtx[0, i] = cfg.init_p2
            V_mtx[1, i] = cfg.init_u2
            V_mtx[2, i] = cfg.init_T2

    U_mtx, F_mtx = setup_euler(cfg, V_mtx)

    U_mtx, V_mtx = solve(cfg, V_mtx, U_mtx, F_mtx, T_vec, dx, tmax, CFL, np.zeros((3, NJ - 2)))

    return U_mtx, V_mtx

def get_new_state(cfg, V_mtx, corr_src):
    U_mtx, F_mtx = setup_euler(cfg, V_mtx)
    T_vec = V_mtx[2,:]
    _, V_mtx = solve(cfg, V_mtx, U_mtx, F_mtx, T_vec, cfg.dx, cfg.dt, cfg.CFL, corr_src)
    return V_mtx

def get_corr_src_term(cfg, old_V_mtx_ref, new_V_mtx_ref):
    old_U_mtx_ref, old_F_mtx_ref = setup_euler(cfg, old_V_mtx_ref)
    old_T_vec_ref = old_V_mtx_ref[2,:]
    new_U_mtx_ref, _ = setup_euler(cfg, new_V_mtx_ref)
    new_U_mtx_num, _ = solve(
        cfg, old_V_mtx_ref, old_U_mtx_ref, old_F_mtx_ref, old_T_vec_ref, cfg.dx, cfg.dt, cfg.CFL, np.zeros(cfg.N_x)
    )
    return new_U_mtx_ref[:,1:-1] - new_U_mtx_num[:,1:-1]

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

    b = T_old[1:-1,1:-1].flatten(order='F')
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
    b += dt * get_src(cfg.x_nodes[1:-1], cfg.y_nodes[1:-1], t_new, alpha).flatten(order='F')

    return b

########################################################################################################################

def simulate_2D(cfg, T_old, t_start, t_end, alpha, get_src, cor_src):
    A = get_A_matrix(cfg)
    time = t_start

    while time < t_end:
        b = get_b_vector(cfg, T_old, time, alpha, get_src)
        b += cor_src.flatten(order='F')
        T_interior_flat = scipy.linalg.solve(A, b)
        T_interior = T_interior_flat.reshape((cfg.N_x, cfg.N_y), order='F')

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
    sigma_corr = np.dot(A, T_new[1:-1, 1:-1].flatten(order='F')) - b
    return np.reshape(sigma_corr, (cfg.N_x, cfg.N_y), order='F')

########################################################################################################################

def main():
    N_xs = [100]#, 9, 27]#, 81, 81*3, 81*9]
    errors = []
    for N_x in N_xs:
        cfg = config.Config(
            use_GPU=config.use_GPU,
            group_name=config.group_name,
            run_name=config.run_names[0][0],
            system=config.systems[0],
            data_tag=config.data_tags[0],
            model_key=config.model_keys[0],
            do_train=False,
            do_test=False,
            N_x=N_x,
            model_type=config.model_type,
        )

        unc_Vs = np.zeros((cfg.N_t, 3, cfg.NJ))
        cor_Vs = np.zeros((cfg.N_t, 3, cfg.NJ))
        ref_Vs = np.zeros((cfg.N_t, 3, cfg.NJ))
        srcs   = np.zeros((cfg.N_t, 3, cfg.N_x))

        time = 0.0

        unc_Vs[0] = get_init_V_mtx(cfg)
        cor_Vs[0] = get_init_V_mtx(cfg)
        ref_Vs[0] = get_init_V_mtx(cfg)

        for i in range(1, cfg.N_t):
            time      = np.around(time + cfg.dt, decimals=10)
            unc_Vs[i] = get_new_state(cfg, unc_Vs[i-1], np.zeros((3, cfg.N_x)))
            _, num_sol = simulate_euler(cfg, time)
            np.testing.assert_allclose(num_sol, unc_Vs[i], rtol=1e-10, atol=1e-10)
            ref_Vs[i] = exact_solver.exact_solver(cfg, time)
            srcs[i]   = get_corr_src_term(cfg, ref_Vs[i-1], ref_Vs[i])
            cor_Vs[i] = get_new_state(cfg, cor_Vs[i-1], srcs[i])
            np.testing.assert_allclose(ref_Vs[i], cor_Vs[i], rtol=1e-10, atol=1e-10)



        fig, axs = plt.subplots(3, 1)
        ylabels = [r"$p$", r"$u$", r"$T$"]
        for j, ax in enumerate(fig.get_axes()):
            axs[j].plot(cfg.x_nodes, unc_Vs[-1, j], 'r-', label='LxF')
            axs[j].plot(cfg.x_nodes, cor_Vs[-1, j], 'b-', label='LxF cor')
            axs[j].plot(cfg.x_nodes, ref_Vs[-1, j], 'g--', label='Exact')
            axs[j].legend()
            axs[j].set_xlabel(r'$x$')
            axs[j].set_ylabel(ylabels[j])
            axs[j].grid()
            ax.label_outer()
        plt.show()

        """
        num_U, num_sol = simulate_euler(cfg, cfg.t_end)
        exact_sol = exact_solver.exact_solver(cfg)

        num_rho = num_sol[0,:] / (num_sol[2,:]*cfg.c_V*(cfg.gamma - 1))
        exact_rho = exact_sol[0,:] / (exact_sol[2,:]*cfg.c_V*(cfg.gamma - 1))

        fig, axs = plt.subplots(4, 1)
        ylabels = [r"$p$", r"$u$", r"$T$", r"$\rho$"]
        for i, ax in enumerate(fig.get_axes()):
            if i == 3:
                break
            axs[i].plot(cfg.x_nodes[1:-1], num_sol[i], 'r-', label='LxF')
            axs[i].plot(cfg.x_nodes[1:-1], exact_sol[i], 'g-', label='Exact')
            axs[i].legend()
            axs[i].set_xlabel(r'$x$')
            axs[i].set_ylabel(ylabels[i])
            axs[i].grid()
            ax.label_outer()
        axs[3].plot(cfg.x_nodes[1:-1], num_rho, 'r-', label='LxF')
        axs[3].plot(cfg.x_nodes[1:-1], exact_rho, 'g-', label='Exact')
        axs[3].legend()
        axs[3].set_xlabel(r'$x$')
        axs[3].set_ylabel(ylabels[3])
        axs[3].grid()
        axs[3].label_outer()
        plt.show()
        """


    """
        alpha = 0.7
        T0 = np.zeros((cfg.N_x + 2, cfg.N_y + 2))
        for i, y in enumerate(cfg.y_nodes):
            for j, x in enumerate(cfg.x_nodes):
                T0[j][i] = cfg.get_T0(x, y, alpha)
        print("T0:", T0)
        print("T0_flattened:", T0.flatten(order='F'))
        T = simulate_2D(cfg, T0, t_start, cfg.t_end, alpha, cfg.get_q_hat, np.zeros((cfg.N_x, cfg.N_y)))
        print("T:", T)
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
    """


########################################################################################################################

if __name__ == "__main__":
    main()

########################################################################################################################