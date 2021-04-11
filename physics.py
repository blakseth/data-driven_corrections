"""
physics.py

Written by Sindre Stenen Blakseth, 2021.

Numerical simulation of the 1D Euler equations and generation of corrective source terms.
"""

########################################################################################################################
# Package imports.

import matplotlib.pyplot as plt
import numpy as np
#import scipy.linalg
#from scipy.sparse import diags
#import sys

########################################################################################################################
# File imports.

import config
import exact_solver
#import util

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

########################################################################################################################

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

########################################################################################################################

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

########################################################################################################################

def get_new_state(cfg, V_mtx, corr_src):
    U_mtx, F_mtx = setup_euler(cfg, V_mtx)
    T_vec = V_mtx[2,:]
    _, V_mtx = solve(cfg, V_mtx, U_mtx, F_mtx, T_vec, cfg.dx, cfg.dt, cfg.CFL, corr_src)
    return V_mtx

########################################################################################################################

def get_corr_src_term(cfg, old_V_mtx_ref, new_V_mtx_ref):
    old_U_mtx_ref, old_F_mtx_ref = setup_euler(cfg, old_V_mtx_ref)
    old_T_vec_ref = old_V_mtx_ref[2,:]
    new_U_mtx_ref, _ = setup_euler(cfg, new_V_mtx_ref)
    new_U_mtx_num, _ = solve(
        cfg, old_V_mtx_ref, old_U_mtx_ref, old_F_mtx_ref, old_T_vec_ref, cfg.dx, cfg.dt, cfg.CFL, np.zeros(cfg.N_x)
    )
    return new_U_mtx_ref[:,1:-1] - new_U_mtx_num[:,1:-1]

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

########################################################################################################################

if __name__ == "__main__":
    main()

########################################################################################################################