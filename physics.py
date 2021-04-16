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
import pickle

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
    return 0.5 * ( F_mtx[:, :-1] + F_mtx[:, 1:] - (dx/dt)*(U_mtx[:, 1:] - U_mtx[:, :-1]) )

def find_wave_speed_Roe_avg(rho_L, u_L, p_L, c_L, rho_R, u_R, p_R, c_R, gamma):
    u_avg = (np.sqrt(rho_L) * u_L + np.sqrt(rho_R) * u_R) / (np.sqrt(rho_L) + np.sqrt(rho_R))
    H_L = 0.5 * u_L ** 2 + c_L ** 2 / (gamma - 1)
    H_R = 0.5 * u_R ** 2 + c_R ** 2 / (gamma - 1)
    H_avg = (np.sqrt(rho_L) * H_L + np.sqrt(rho_R) * H_R) / (np.sqrt(rho_L) + np.sqrt(rho_R))
    c_avg = np.sqrt((gamma - 1) * (H_avg - 0.5 * u_avg ** 2))

    SL = min(u_L - c_L, u_avg - c_avg)
    SR = max(u_R + c_R, u_avg + c_avg)
    SM = (p_R - p_L + rho_L*u_L*(SL - u_L) - rho_R*u_R*(SR - u_R)) / (rho_L*(SL - u_L) - rho_R*(SR - u_R))

    return SL, SM, SR

def HLL_flux(U_mtx, V_mtx, F_mtx, c_vec, gamma):
    F_est = np.zeros((F_mtx.shape[0], F_mtx.shape[1] - 1))
    for i in range(U_mtx.shape[1] - 1):
        rho_L = U_mtx[0,i]
        u_L   = V_mtx[1,i]
        p_L   = V_mtx[0,i]
        c_L   = c_vec[i]

        rho_R = U_mtx[0,i+1]
        u_R = V_mtx[1,i+1]
        p_R = V_mtx[0,i+1]
        c_R = c_vec[i+1]

        SL, SM, SR = find_wave_speed_Roe_avg(rho_L, u_L, p_L, c_L, rho_R, u_R, p_R, c_R, gamma)

        if SL >= 0.0: # Right-going supersonic flow.
            F_est[:,i] = F_mtx[:,i]
        elif SR <= 0.0: # Left-going supersonic flow.
            F_est[:,i] = F_mtx[:,i+1]
        else: # Subsonic flow.
            f_HLL = (SR*F_mtx[:,i] - SL*F_mtx[:,i+1] + SL*SR*(U_mtx[:,i+1] - U_mtx[:, i])) / (SR - SL)
            F_est[:,i] = f_HLL
    return F_est

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
    for energy in e:
        if not energy > 0:
            print("WARNING: Zero energy in setup_euler")
    U_mtx[0,:] = V_mtx[0,:] / ((cfg.gamma - 1)*e)
    U_mtx[1,:] = U_mtx[0,:] * V_mtx[1,:]
    U_mtx[2,:] = U_mtx[0,:]*e + 0.5*U_mtx[0,:]*V_mtx[1,:]**2

    F_mtx = find_flux(U_mtx, V_mtx)

    return U_mtx, F_mtx

########################################################################################################################

def solve(cfg, V_mtx, U_mtx, F_mtx, T_vec, dx, tmax, CFL, corr_src, solver_type):

    counter = 0
    t = 0.0
    dt = cfg.dt
    c_vec = find_c(cfg, T_vec)
    u_vec = V_mtx[1, :]

    while True:
        check_valid_dt(cfg, u_vec, c_vec, CFL, dx)
        t = np.around(t + cfg.dt, decimals=10)
        print("t =", t)
        print("tmax = ", tmax)

        if solver_type == 'LxF':
            F_est = LxF_flux(U_mtx, F_mtx, dt, dx)
        elif solver_type == 'HLL':
            F_est = HLL_flux(U_mtx, V_mtx, F_mtx, c_vec, cfg.gamma)
        else:
            raise Exception("Invalid solver type.")

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

def simulate_euler(cfg, t_end, solver_type):
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

    U_mtx, V_mtx = solve(cfg, V_mtx, U_mtx, F_mtx, T_vec, dx, tmax, CFL, np.zeros((3, NJ - 2)), solver_type)

    return U_mtx, V_mtx

########################################################################################################################

def get_new_state(cfg, old_V_mtx, corr_src, solver_type):
    for T in old_V_mtx[2,:]:
        if not T > 0:
            print("Non-positive T, get_new_state")
    old_U_mtx, old_F_mtx = setup_euler(cfg, old_V_mtx)
    old_T_vec = old_V_mtx[2,:]
    _, V_mtx = solve(cfg, old_V_mtx, old_U_mtx, old_F_mtx, old_T_vec, cfg.dx, cfg.dt, cfg.CFL, corr_src, solver_type)
    return V_mtx

########################################################################################################################

def get_corr_src_term(cfg, old_V_mtx_ref, new_V_mtx_ref, solver_type):
    old_U_mtx_ref, old_F_mtx_ref = setup_euler(cfg, old_V_mtx_ref)
    old_T_vec_ref = old_V_mtx_ref[2,:]
    new_U_mtx_ref, _ = setup_euler(cfg, new_V_mtx_ref)
    new_U_mtx_num, _ = solve(
        cfg, old_V_mtx_ref, old_U_mtx_ref, old_F_mtx_ref, old_T_vec_ref, cfg.dx, cfg.dt, cfg.CFL, np.zeros(cfg.N_x), solver_type
    )
    return (new_U_mtx_ref[:,1:-1] - new_U_mtx_num[:,1:-1])

########################################################################################################################

def main():

    cfg = config.Config(
        use_GPU=config.use_GPU,
        group_name=config.group_name,
        run_name=config.run_names[0][0],
        system=config.systems[0],
        data_tag=config.data_tags[0],
        model_key=config.model_keys[0],
        do_train=False,
        do_test=False,
        N_x=100,
        model_type=config.model_type,
    )
    print("alphas:", cfg.alphas)
    for time in [0.0001, 0.001, 0.005, 0.01, 0.03, 0.07, 0.12, 2.0]:
        V = exact_solver.exact_solver(cfg, time)
        fig, axs = plt.subplots(4, 1)
        fig.suptitle("Time: " + str(time))
        ylabels = [r"$p$", r"$u$", r"$T$"]
        for j in range(3):
            axs[j].plot(cfg.x_nodes, V[j])
            axs[j].legend()
            axs[j].set_xlabel(r'$x$')
            axs[j].set_ylabel(ylabels[j])
            axs[j].grid()
            axs[j].label_outer()
        axs[3].plot(cfg.x_nodes, V[0] / (cfg.c_V * (cfg.gamma - 1) * V[2]))
        axs[3].set_xlabel(r'$x$')
        axs[3].set_ylabel(r'$\rho$')
        axs[3].grid()
        axs[3].label_outer()
    plt.show()


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

        unc_Vs = np.zeros((cfg.alphas.shape[0], cfg.N_t, 3, cfg.NJ))
        HLL_Vs = np.zeros((cfg.alphas.shape[0], cfg.N_t, 3, cfg.NJ))
        cor_Vs = np.zeros((cfg.alphas.shape[0], cfg.N_t, 3, cfg.NJ))
        ref_Vs = np.zeros((cfg.alphas.shape[0], cfg.N_t, 3, cfg.NJ))
        srcs   = np.zeros((cfg.alphas.shape[0], cfg.N_t, 3, cfg.N_x))

        maxs = []
        mins = []
        twos = []

        for a, alpha in enumerate(cfg.alphas):
            print("alpha:", alpha)
            max_up = np.zeros(3)
            max_down = np.zeros(3)
            two = np.zeros(3)

            cfg.init_u1 = cfg.init_u2 = 0.1 * alpha

            unc_Vs[a][0] = get_init_V_mtx(cfg)
            #print("unv_Vs[a][0]", unc_Vs[a][0])
            HLL_Vs[a][0] = get_init_V_mtx(cfg)
            cor_Vs[a][0] = get_init_V_mtx(cfg)
            ref_Vs[a][0] = get_init_V_mtx(cfg)

            time = 0.0

            for i in range(1, cfg.N_t - 1):
                flag = False
                time      = np.around(time + cfg.dt, decimals=10)
                #print("i:", i)
                unc_Vs[a][i] = get_new_state(cfg, unc_Vs[a][i-1], np.zeros((3, cfg.N_x)), 'LxF')
                #print("unc_Vs[a][i-1]:", unc_Vs[a][i-1])
                HLL_Vs[a][i] = get_new_state(cfg, HLL_Vs[a][i - 1], np.zeros((3, cfg.N_x)), 'HLL')
                _, num_sol = simulate_euler(cfg, time, 'LxF')
                np.testing.assert_allclose(num_sol, unc_Vs[a][i], rtol=1e-10, atol=1e-10)
                ref_Vs[a][i] = np.around(exact_solver.exact_solver(cfg, time), 10)
                srcs[a][i]   = np.around(get_corr_src_term(cfg, ref_Vs[a][i-1], ref_Vs[a][i], 'LxF'), 10)
                cor_Vs[a][i] = get_new_state(cfg, cor_Vs[a][i-1], srcs[a][i], 'LxF')
                np.testing.assert_allclose(ref_Vs[a][i], cor_Vs[a][i], rtol=1e-9, atol=1e-9)

                if np.around(max_up[0],10) == np.around(0.0, 10) and np.around(np.amin(srcs[a][i]), 10) < np.around(0.0, 10):
                    max_up = np.amax(srcs[a][i], axis=1)
                    max_down = np.amin(srcs[a][i], axis=1)
                    print("Set max_up and max_down")
                    print("max_up set to", max_up)
                    print("max_down set to", max_down)
                elif np.around(two[0],10) == np.around(0.0, 10) and np.around(np.amin(srcs[a][i]), 10) == np.around(0.0, 10):
                    two = np.amax(srcs[a][i], axis=1)
                    print("two set to", two)
                    print("Set two")
                for j in range(3):
                    if np.around(np.amax(srcs[a][i][j]),7) not in [np.around(max_up[j],7), np.around(two[j],7)]:
                        print("WARNING3")
                        print("max_up[j]:", max_up[j])
                        print("two[j]:", two[j])
                        print("srcs[a][i]:", srcs[a][i])
                        print("ref_Vs[a][i]:", ref_Vs[a][i])
                        print("j=", j)
                        flag = True
                    if np.around(np.amin(srcs[a][i][j]),7) not in [np.around(max_down[j],7), np.around(0.0,7)]:
                        print("WARNING4")
                        print("max_down[j]:", max_down[j])
                        print("srcs[a][i]:", srcs[a][i])
                        print("ref_Vs[a][i]:", ref_Vs[a][i])
                        print("j=", j)
                        flag = True



                for j in range(3):
                    if (not np.argmax(srcs[a][i][j]) + 1 == np.argmin(srcs[a][i][j])) and (not (np.around(srcs[a][i][j][np.argmax(srcs[a][i][j])], 5) == np.around(srcs[a][i][j][np.argmax(srcs[a][i][j]) + 1], 5) or np.around(srcs[a][i][j][np.argmax(srcs[a][i][j])], 5) == np.around(srcs[a][i][j][np.argmax(srcs[a][i][j]) - 1], 5) )):
                        print("WARNING1")
                        print("src:", srcs[a][i])
                        print("ref:", ref_Vs[a][i])
                        print("j=", j)
                        flag = True

                diff_V = np.abs(ref_Vs[a, i, :, 1:] - ref_Vs[a, i, :, :-1])
                print("ref_Vs.shape", ref_Vs.shape)
                print("diff_V-shape", diff_V.shape)
                for j in range(3):
                    if np.argmax(diff_V[j]) != 0 and (not np.argmax(diff_V[j]) == np.argmax(srcs[a][i][j])+1) and (not (np.around(srcs[a][i][j][np.argmax(srcs[a][i][j])], 5) == np.around(srcs[a][i][j][np.argmax(srcs[a][i][j]) + 1], 5) or np.around(srcs[a][i][j][np.argmax(srcs[a][i][j])], 5) == np.around(srcs[a][i][j][np.argmax(srcs[a][i][j]) - 1], 5) )):
                        print("WARNING2")
                        print("index1:", np.argmax(diff_V[j]))
                        print("index2:", np.argmax(srcs[a][i][j]))
                        print("src:", srcs[a][i])
                        print("diff_V:", diff_V)
                        print("ref_V:", ref_Vs[a][i])
                        print("j=", j)
                        flag = True

                if flag:
                    raise Exception

            maxs.append(max_up)
            mins.append(max_down)
            twos.append(two)

        my_dict = {
            "maxs": np.asarray(maxs),
            "mins": np.asarray(mins),
            "twos": np.asarray(twos)
        }

        print("my_dict", my_dict)

        pickle.dump(my_dict, open("targets.pkl", "wb"))

        loaded_dict = pickle.load(open("targets.pkl", "rb"))

        print("loaded_dict:", loaded_dict)

        fig, axs = plt.subplots(4, 1)
        ylabels = [r"$p$", r"$u$", r"$T$"]
        for j in range(3):
            axs[j].plot(cfg.x_nodes, unc_Vs[-1, j], 'r-', label='LxF')
            axs[j].plot(cfg.x_nodes, HLL_Vs[-1, j], 'y-', label='HLL')
            axs[j].plot(cfg.x_nodes, cor_Vs[-1, j], 'b-', label='LxF cor')
            axs[j].plot(cfg.x_nodes, ref_Vs[-1, j], 'g--', label='Exact')
            axs[j].legend()
            axs[j].set_xlabel(r'$x$')
            axs[j].set_ylabel(ylabels[j])
            axs[j].grid()
            axs[j].label_outer()
        axs[3].plot(cfg.x_nodes, unc_Vs[-1, 0] / (cfg.c_V * (cfg.gamma - 1) * unc_Vs[-1,2]), 'r-', label='LxF')
        axs[3].plot(cfg.x_nodes, HLL_Vs[-1, 0] / (cfg.c_V * (cfg.gamma - 1) * HLL_Vs[-1,2]), 'y-', label='HLL')
        axs[3].plot(cfg.x_nodes, cor_Vs[-1, 0] / (cfg.c_V * (cfg.gamma - 1) * cor_Vs[-1,2]), 'b-', label='LxF cor')
        axs[3].plot(cfg.x_nodes, ref_Vs[-1, 0] / (cfg.c_V * (cfg.gamma - 1) * ref_Vs[-1,2]), 'g--', label='Exact')
        axs[3].legend()
        axs[3].set_xlabel(r'$x$')
        axs[3].set_ylabel(r'$\rho$')
        axs[3].grid()
        axs[3].label_outer()
        plt.show()

        for alpha in cfg.test_alphas:
            cfg.init_u2 = alpha
            plt.figure()
            plt.plot(cfg.x_nodes, exact_solver.exact_solver(cfg, cfg.t_end)[0])
        plt.show()

########################################################################################################################

if __name__ == "__main__":
    main()

########################################################################################################################