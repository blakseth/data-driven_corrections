"""
exact_solver.py

Written by E. F. Toro in FORTRAN, 1999.
Translated to Python by Sindre Stenen Blakseth, 2021.

Exact solver of the Riemann problem for the time dependent one dimensional Euler equations for an ideal gas.
"""

########################################################################################################################
# Package imports.

import numpy as np

########################################################################################################################
# Helper functions.

def GUESSP(DL, UL, PL, CL, DR, UR, PR, CR, G1, G3, G4, G5, G6, G7):
    QUSER = 2.0

    CUP  = 0.25 * (DL + DR) * (CL + CR)
    PPV  = 0.5 * (PL + PR) + 0.5 * (UL - UR) * CUP
    PPV  = max(0.0, PPV)
    PMIN = min(PL, PR)
    PMAX = max(PL, PR)
    QMAX = PMAX / PMIN

    if QMAX <= QUSER and (PMIN <= PPV <= PMAX):
        # Select PVRS Riemann solver.
        PM = PPV
    elif PPV < PMIN:
        # Select Two-Rarefaction Riemann solver.
        PQ  = (PL/PR)**G1
        UM  = (PQ*UL/CL + UR/CR + G4*(PQ - 1.0)) / (PQ/CL + 1.0/CR)
        PTL = 1.0 + G7*(UL - UM)/CL
        PTR = 1.0 + G7*(UM - UR)/CR
        PM  = 0.5 * (PL*PTL**G3 + PR*PTR ** G3)
    else:
        # Select Two-Shock Riemann solver with PVRS as estimate.
        GEL = np.sqrt((G5/DL) / (G6*PL + PPV))
        GER = np.sqrt((G5/DR) / (G6*PR + PPV))
        PM  = (GEL*PL + GER*PR - (UR - UL)) / (GEL + GER)
    return PM

def PREFUN(P, DK, PK, CK, G1, G2, G4, G5, G6):
    if P <= PK: # Rarefaction wave
        PRATIO = P/PK
        F      = G4*CK*(PRATIO**G1 - 1.0)
        FD     = (1.0/(DK*CK)) * PRATIO**(-G2)
    else: # Shock wave
        AK  = G5/DK
        BK  = G6*PK
        QRT = np.sqrt(AK/(BK + P))
        F   = (P - PK)*QRT
        FD  = ( 1.0 - 0.5*(P - PK) / (BK + P) )*QRT
    return F, FD

def STARPU(DL, UL, PL, CL, DR, UR, PR, CR, G1, G2, G3, G4, G5, G6, G7, PSCALE):
    TOLPRE = 1e-6
    NRITER = 20
    FR = FL = P = np.inf

    # Guessed value of P is computed.
    PSTART = GUESSP(DL, UL, PL, CL, DR, UR, PR, CR, G1, G3, G4, G5, G6, G7)

    # P is computed.
    POLD  = PSTART
    UDIFF = UR - UL
    for i in range(NRITER):
        FL, FLD = PREFUN(POLD, DL, PL, CL, G1, G2, G4, G5, G6)
        FR, FRD = PREFUN(POLD, DR, PR, CR, G1, G2, G4, G5, G6)

        P = POLD - (FL + FR + UDIFF) / (FLD + FRD)
        CHANGE = 2.0 * np.abs((P - POLD) / (P + POLD))
        print("i: " + str(i) + ", change: " + str(CHANGE))
        if CHANGE <= TOLPRE:
            break

        if P < 0.0:
            P = TOLPRE
        POLD = P

        if i == NRITER - 1:
            print("WARNING: Divergence in Newton-Raphson iteration.")

    # Velocity is computed.
    U = 0.5 * (UL + UR + FR - FL)

    assert P < np.inf and FL < np.inf and FR < np.inf

    SCALEDP = P/PSCALE
    print("Pressure: " + str(SCALEDP) + ", velocity: " + str(U))
    return SCALEDP, U

def SAMPLE(PM, UM, S, DL, UL, PL, CL, DR, UR, PR, CR, GAMMA, G1, G2, G3, G4, G5, G6, G7):
    if S <= UM: # Sampling point lies to the left of the contact discontinuity.
        if PM <= PL: # Left rarefaction.
            SHL = UL - CL
            if S <= SHL: # Sampled point is left data state.
                D = DL
                U = UL
                P = PL
            else:
                CML = CL * (PM / PL) ** G1
                STL = UM - CML

                if S > STL: # Sampled point is Star Left state.
                    D = DL * (PM / PL) ** (1.0 / GAMMA)
                    U = UM
                    P = PM
                else: # Sampled point is inside left fan.
                    U = G5 * (CL + G7 * UL + S)
                    C = G5 * (CL + G7 * (UL - S))
                    D = DL * (C / CL) ** G4
                    P = PL * (C / CL) ** G3
        else: # Left shock.
            PML = PM / PL
            SL  = UL - CL * np.sqrt(G2 * PML + G1)

            if S <= SL: # Sampled point is left data state.
                D = DL
                U = UL
                P = PL
            else: # Sampled point is Star Left state.
                D = DL * (PML + G6) / (PML * G6 + 1.0)
                U = UM
                P = PM
    else: # Sampling point lies to the right of the contact discontinuity.
        if PM > PR: # Right shock.
            PMR = PM / PR
            SR = UR + CR * np.sqrt(G2 * PMR + G1)

            if S >= SR: # Sampled point is right data state.
                D = DR
                U = UR
                P = PR
            else: # Sampled point is Star Right state.
                D = DR * (PMR + G6) / (PMR * G6 + 1.0)
                U = UM
                P = PM
        else: # Right rarefaction.
            SHR = UR + CR

            if S >= SHR: # Sampled point is right data state.
                D = DR
                U = UR
                P = PR
            else:
                CMR = CR * (PM / PR) ** G1
                STR = UM + CMR

                if S <= STR: # Sampled point is Star Right state.
                    D = DR * (PM / PR) ** (1.0 / GAMMA)
                    U = UM
                    P = PM
                else: # Sampled point is inside left fan.
                    U = G5 * (-CR + G7 * UR + S)
                    C = G5 * (CR - G7 * (UR - S))
                    D = DR * (C / CR) ** G4
                    P = PR * (C / CR) ** G3

    return D, U, P



########################################################################################################################
# Exact solver.

def exact_solver(cfg):
    # Input variables:
    # DOMLEN : Domain length
    # DIAPH1 : Position of diapraghm 1
    # CELLS  : Number of computing cells
    # GAMMA  : Ratio of specific heats
    # TIMEOU : Output time
    # DL     : Initial density  on left  state
    # UL     : Initial velocity on left  state
    # PL     : Initial pressure on left  state
    # DR     : Initial density  on right state
    # UR     : Initial velocity on right state
    # PR     : Initial pressure on right state
    # PSCALE : Normalizing constant

    # Read input variables from config.
    DOMLEN = cfg.x_b - cfg.x_a
    DIAPH1 = cfg.x_split
    CELLS  = cfg.N_x
    GAMMA  = cfg.gamma
    TIMEOU = cfg.t_end
    DL     = cfg.init_rho1
    UL     = cfg.init_u1
    PL     = cfg.init_p1
    DR     = cfg.init_rho2
    UR     = cfg.init_u2
    PR     = cfg.init_p2
    CV     = cfg.c_V
    PSCALE = 1.0

    # Compute gamma-related constants.
    G1 = (GAMMA - 1.0) / (2.0*GAMMA)
    G2 = (GAMMA + 1.0) / (2.0*GAMMA)
    G3 = 2.0*GAMMA / (GAMMA - 1.0)
    G4 = 2.0 / (GAMMA - 1.0)
    G5 = 2.0 / (GAMMA + 1.0)
    G6 = (GAMMA - 1.0) / (GAMMA + 1.0)
    G7 = (GAMMA - 1.0) / 2.0
    G8 = GAMMA - 1.0

    # Compute sound speeds.
    CL = np.sqrt(GAMMA * PL / DL)
    CR = np.sqrt(GAMMA * PR / DR)

    # Testing pressure positivity condition.
    if G4*(CL + CR) <= UR - UL:
        raise Exception("Pressure positivity condition not satisfied. Program stopped!")

    # Exact solution for pressure and velocity in star region is found.
    PM, UM = STARPU(DL, UL, PL, CL, DR, UR, PR, CR, G1, G2, G3, G4, G5, G6, G7, PSCALE)

    DX = DOMLEN / float(CELLS)

    # Complete solution EXACT_SOL at time TIMEOU is found.
    EXACTV = np.zeros((3, CELLS))
    for i in range(CELLS):
        XPOS = (i + 0.5)*DX
        S    = (XPOS - DIAPH1)/TIMEOU

        # Solution at point (X,T) = (XPOS - DIAPH1, TIMEOU) is found.
        DS, US, PS = SAMPLE(PM, UM, S, DL, UL, PL, CL, DR, UR, PR, CR, GAMMA, G1, G2, G3, G4, G5, G6, G7)

        # Store exact solution.
        EXACTV[0,i] = (PS/PSCALE)
        EXACTV[1,i] = US
        EXACTV[2,i] = PS / (DS*CV*G8)

    return EXACTV

########################################################################################################################
