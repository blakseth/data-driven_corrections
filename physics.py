"""
physics.py

Written by Sindre Stenen Blakseth, 2021.

Numerical simulation of the 1D heat equation and generation of corrective source terms.
"""

########################################################################################################################
# Package imports.

import numpy as np
from scipy.sparse import diags

########################################################################################################################
# TDMA solver for tri-diagonal systems.

def tdma(low_diag, main_diag, high_diag, rhs, N):
    x = np.zeros(N)
    # Forward sweep
    for i in range(1, N):
        w = low_diag[i-1]/main_diag[i-1]
        main_diag[i] -= w*high_diag[i-1]
        rhs[i]       -= w*rhs[i-1]
    # Backward sweep
    x[-1] = rhs[-1] / main_diag[-1]
    for i in range(N-2, -1, -1):
        x[i] = (rhs[i] - high_diag[i]*x[i+1]) / main_diag[i]
    return x

########################################################################################################################
# Physics simulation.

def simulate(nodes, faces, T0, T_a, T_b, get_k, get_cV, rho, A, get_source, dt, t_end, steady):
    assert steady or T0.shape[0] == nodes.shape[0]
    assert faces.shape[0] == nodes.shape[0] - 1
    assert steady or T0 is not None

    N = faces.shape[0] - 1

    dx_int = faces[1:] - faces[:-1]
    dx_half_int = nodes[1:] - nodes[:-1]

    k_nodes = get_k(nodes)
    cV_nodes = get_cV(nodes)

    alpha_nodes = k_nodes / (rho * cV_nodes)
    alpha_half_int = 2 * alpha_nodes[:-1] * alpha_nodes[1:] / (alpha_nodes[:-1] + alpha_nodes[1:])
    alpha_half_int[0] = alpha_nodes[0]
    alpha_half_int[-1] = alpha_nodes[-1]

    sigma = get_source(nodes[1:-1]) / (rho * cV_nodes[1:-1])

    if T0 is not None:
        T = T0[1:-1]
    elif steady:
        T = np.zeros(nodes.shape[0] - 2)
    else:
        raise Exception("Missing initial condition.")

    print("Alpha:", alpha_nodes)
    print("Alpha face:", alpha_half_int)
    print("Sigma:", sigma)

    if steady:
        # Define coefficient matrix of linear system.
        diag = (alpha_half_int[1:] / dx_half_int[1:] + alpha_half_int[:-1] / dx_half_int[:-1])/dx_int  # Main diagonal.
        off_diag_up = -alpha_half_int[1:-1] / (dx_int[1:] * dx_half_int[1:-1])  # Off-diagonal directly above main diagonal.
        off_diag_dn = -alpha_half_int[1:-1] / (dx_int[:-1] * dx_half_int[1:-1]) # Off-diagonal directly below main diagonal.

        # Define RHS vector.
        b = sigma
        b[0] += alpha_half_int[0] * T_a / (dx_int[0] * dx_half_int[0])
        b[-1] += alpha_half_int[-1] * T_b / (dx_int[-1] * dx_half_int[-1])

        # Solve linear system.
        T = tdma(off_diag_dn, diag, off_diag_up, b, N)

    if not steady:
        # Define coefficient matrix of linear system.
        diag = np.ones(N) + dt * (
                    alpha_half_int[1:] / dx_half_int[1:] + alpha_half_int[:-1] / dx_half_int[:-1])/dx_int  # Main diagonal.
        off_diag_up = -dt * alpha_half_int[1:-1] / (dx_int[1:] * dx_half_int[1:-1])  # Off-diagonal directly above main diagonal.
        off_diag_dn = -dt * alpha_half_int[1:-1] / (dx_int[:-1] * dx_half_int[1:-1])  # Off-diagonal directly below main diagonal.

        # Initialize time.
        time = 0

        while time < t_end:
            # Increment time.
            time += dt

            # Define RHS vector.
            b = T + dt * sigma
            b[0] += alpha_half_int[0] * dt * T_a / (dx_int[0] * dx_half_int[0])
            b[-1] += alpha_half_int[-1] * dt * T_b / (dx_int[-1] * dx_half_int[-1])

            # Solve linear system.
            new_T = tdma(off_diag_dn.copy(), diag.copy(), off_diag_up.copy(), b, N)
            np.set_printoptions(precision=12)
            T = new_T

    T_including_boundary = np.zeros(N + 2)
    T_including_boundary[0] = T_a
    T_including_boundary[1:-1] = T
    T_including_boundary[-1] = T_b

    return T_including_boundary

########################################################################################################################
# Corrective source term.

def get_corrective_src_term(nodes, faces, T_ref_new, T_ref_old, T_a, T_b, get_k, get_cV, rho, area, get_source, dt, steady):
    assert T_ref_new.shape[0] == nodes.shape[0]
    assert T_ref_old is None or T_ref_old.shape[0] == nodes.shape[0]
    assert faces.shape[0]   == nodes.shape[0] - 1

    N = faces.shape[0] - 1

    dx_int = faces[1:] - faces[:-1]
    dx_half_int = nodes[1:] - nodes[:-1]

    k_nodes = get_k(nodes)
    cV_nodes = get_cV(nodes)

    alpha_nodes = k_nodes / (rho * cV_nodes)
    alpha_half_int = 2 * alpha_nodes[:-1] * alpha_nodes[1:] / (alpha_nodes[:-1] + alpha_nodes[1:])
    alpha_half_int[0] = alpha_nodes[0]
    alpha_half_int[-1] = alpha_nodes[-1]

    sigma = get_source(nodes[1:-1]) / (rho * cV_nodes[1:-1])

    A = None
    b = None

    if steady:
        # Define coefficient matrix of linear system.
        diag = (alpha_half_int[1:] / dx_half_int[1:] + alpha_half_int[:-1] / dx_half_int[:-1])/dx_int  # Main diagonal.
        off_diag_up = -alpha_half_int[1:-1] / (dx_int[1:] * dx_half_int[1:-1])  # Off-diagonal directly above main diagonal.
        off_diag_dn = -alpha_half_int[1:-1] / (dx_int[:-1] * dx_half_int[1:-1]) # Off-diagonal directly below main diagonal.
        A = diags([off_diag_dn, diag, off_diag_up], [-1, 0, 1]).toarray()  # The coefficient matrix.

        # Define RHS vector.
        b = sigma
        b[0] += alpha_half_int[0] * T_a / (dx_int[0] * dx_half_int[0])
        b[-1] += alpha_half_int[-1] * T_b / (dx_int[-1] * dx_half_int[-1])

    if not steady:
        # Define coefficient matrix of linear system.
        diag = np.ones(N) + dt * (
                    alpha_half_int[1:] / dx_half_int[1:] + alpha_half_int[:-1] / dx_half_int[:-1])/dx_int  # Main diagonal.
        off_diag_up = -dt * alpha_half_int[1:-1] / (dx_int[1:] * dx_half_int[1:-1])  # Off-diagonal directly above main diagonal.
        off_diag_dn = -dt * alpha_half_int[1:-1] / (dx_int[:-1] * dx_half_int[1:-1])  # Off-diagonal directly below main diagonal.
        A = diags([off_diag_dn, diag, off_diag_up], [-1, 0, 1]).toarray()  # The coefficient matrix.

        # Define  RHS vector.
        b = T_ref_old[1:-1] + dt * sigma
        b[0] += alpha_half_int[0] * dt * T_a / (dx_int[0] * dx_half_int[0])
        b[-1] += alpha_half_int[-1] * dt * T_b / (dx_int[-1] * dx_half_int[-1])

    # Calculate corrective source term.
    sigma_corr = np.dot(A, T_ref_new[1:-1]) - b
    if steady:
        return sigma_corr
    else:
        return sigma_corr / dt

########################################################################################################################

def main():
    pass

########################################################################################################################

if __name__ == "__main__":
    main()

########################################################################################################################