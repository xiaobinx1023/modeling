from math import pi
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

def cp_pll_ode(t, y, params, ip):
    """
    ODE for the CP-PLL loop filter + phase error, with constant ip over this sub-interval.
    y = [theta_e, v_c1, v_c2]
    ip = current (A) [could be +I_p, -I_p, or 0, constant in sub-interval]
    v3--------------||-------
        |       |
        R1      R2
    v1--|       |----------v2
        C1      C2
    """
    theta_e, vc1, vc2, vc3, vc4 = y
    C1 = params["C1"]
    C2 = params["C2"]
    R1 = params["R1"]
    R2 = params["R2"]
    Kvco = params["Kvco"]
    Kvdd = params["Kvdd"]
    wref = params["wref"]
    wfr = params["wfr"]
    wp = params["wp"]
    Cff = params["Cff"]
    A = params["A"]
    # ODEs
    # dtheta_dt = wref - (Kvco * vc2 + 20e9) / 160
    dtheta_dt = wref - (Kvco * vc2 + wfr - Kvdd * vc4) / 160
    dvc1_dt = vc3 / R1 / C1 - vc1 / R1 / C1
    dvc2_dt = vc3 / R2 / C2 - vc2 / R2 / C2
    dvc3_dt = ip / Cff - vc3 * (1 / R1 + 1 / R2) / Cff + vc1 / R1 / Cff + vc2 / R2 / Cff + A * wp * vc1 - A * wp * 0.375 - wp * vc4
    dvc4_dt =  A * wp * vc1 - A * wp * 0.375  - wp * vc4

    return [dtheta_dt, dvc1_dt, dvc2_dt, dvc3_dt, dvc4_dt]

def boundary_equ(tp, current_state, params, ip):
    """
    return the theta_error at tp
    """
    sol = solve_ivp(cp_pll_ode, 
                    (0, tp), 
                    y0=current_state, 
                    args=(params, ip), 
                    # rtol=1e8, 
                    # atol=1e-10,
                    method='RK45')
    
    theta_e_tp = sol.y[0, -1]
    theta_i_0 = sol.y[0, 0]
    return theta_e_tp - theta_i_0 - params["wref"] * tp + 2 * np.pi
def find_tp(current_state, params, ip, tper):
    def f_tp(tp):
        return boundary_equ(tp, current_state, params, ip)
    brackets = [
        (1e-15, tper),
        (1e-15, tper * 10),
        (1e-15, tper * 100)
    ]
    for bracket in brackets:
        try:
            res = root_scalar(f_tp, bracket=bracket, method='brentq')
            if res.converged:
                return res.root
        except:
            continue
    print("warning: Could not find the solution for tp")
    return tper/2

def boundary_T_equ(t_p, T, state_tp, params):
    """

    """
    sol_off = solve_ivp(
        cp_pll_ode,
        (t_p, T),
        state_tp,
        args=(params, 0.0),
        # rtol=1e-8,
        # atol=1e-10,
        dense_output=False
    )
    final_state = sol_off.y[:, -1]
    theta_e_T = final_state[0]
    return theta_e_T - params["wref"] * T + 2 * np.pi + params['wfr'] * t_p

def find_T(t_p, current_state, params, bracket=(0.0, 1e-6)):
    theta_i_tp = params["wref"] * t_p
    Tref_ = (2 * np.pi - theta_i_tp) / params["wref"]
    def f_T(T):
        return boundary_T_equ(t_p, T, current_state, params)
    res = root_scalar(f_T, bracket=bracket)
    if res.converged:
        return np.min([res.root, Tref_])
    return bracket[0]

def simulate_one_cycle(init_cond, params, t_per):
    """
    """
    theta_e0 = init_cond[0]
    
    # Decide sign => direction of IP
    if theta_e0 > 0:
        ip_on = +params["I_p"]
        t_p = find_tp(init_cond, params, ip_on, t_per)
        print(f"t_p = {t_p}")
    else:
        ip_on = -params["I_p"]
        t_p = - theta_e0 / params["wref"]
    if t_p > t_per / 2:
        t_p = t_per / 2

    # Integrate sub-interval [0..t_p] with CP on
    # sol_on = solve_ivp(cp_pll_ode, (0, t_p), init_cond, args=(params, ip_on),rtol=1e-8, atol=1e-10, dense_output=True)
    sol_on = solve_ivp(cp_pll_ode, (0, t_p), init_cond, args=(params, ip_on), dense_output=True)

    state_tp = sol_on.y[:, -1]
    T_calc = find_T(t_p, state_tp, params)
    print(f"T_calc is {T_calc}")

    # sol_off = solve_ivp(cp_pll_ode, (t_p, T_calc), state_tp, args=(params, 0.0),rtol=1e-8, atol=1e-10,dense_output=True)
    sol_off = solve_ivp(cp_pll_ode, (t_p, T_calc), state_tp, args=(params, 0.0),dense_output=True)

    state_end = sol_off.y[:, -1]

    # Combine times and states
    t_on = sol_on.t
    y_on = sol_on.y
    t_off = sol_off.t
    y_off = sol_off.y

    # Avoid duplication of last point from sol_on
    if t_off[0] == t_on[-1]:
        t_off = t_off[1:]
        y_off = y_off[:, 1:]

    t_full = np.concatenate((t_on, t_off))
    y_full = np.concatenate((y_on, y_off), axis=1)
    
    return state_end, t_full, y_full, t_full[-1]



def run_transient(init_cond, params, T_cycle, n_cycles=1000):
    """
    Run n_cycles in a row, concatenating time + states for a full transient.
    """
    y_current = init_cond
    t_global = []
    y_global = []
    current_t_offset = 0.0
    print(f"current_state = {y_current}")
    for cycle_index in range(n_cycles):
        state_end, t_full, y_full, t_p = simulate_one_cycle(y_current, params, T_cycle)
        
        # Shift the time array by current_t_offset so it's global
        t_shifted = t_full + current_t_offset
        
        # Store
        if cycle_index == 0:
            t_global = t_shifted
            y_global = y_full
        else:
            # remove the duplicate start
            t_shifted = t_shifted[1:]
            y_full = y_full[:, 1:]
            t_global = np.concatenate((t_global, t_shifted))
            y_global = np.concatenate((y_global, y_full), axis=1)
        
        # Prepare for next cycle
        y_current = state_end
        current_t_offset += t_p

    return t_global, y_global
# -------------------- Example usage --------------------
if __name__ == "__main__":
    # Parameters
    params = {

        "Kcp": 100e-6/2/np.pi,  # Charge pump current [A]
        "Kvdd": 20e9 * 2 * np.pi,
        "Cff" : 150e-15,
        "A": 100, # error amp gain
        "wp": 10e3 * 2 * np.pi,
        "C1": 65e-12,
        "C2": 5e-12,
        "R1": 20e3,
        "R2": 1e3,
        "Kvco": 3e9 * 2 * np.pi,      # [rad/(s*V)]
        "wref": 100e6 * 2* np.pi,      # reference freq [rad/s]
        "I_p": 100e-6,       # CP current [A]
        "wfr": 14.875e9 * 2 * np.pi
    }

    # initial conditions: y = [theta_e, vc1, vc2]
    y0 = [-1, 0.0, 0.0, 0.0 , 0.0]     # start with some positive phase error
    T_minus = 1e-8     # total interval for one cycle, for example

    # Simulate one "cycle" from t=0..T_minus
    # final_state, tp_found, sol_on, sol_off = simulate_one_cycle(y0, params, T_minus)


    t_global, y_global = run_transient(y0, params, T_minus,50)

    # y_global.shape => (3, # of time samples)
    # Plot or analyze
    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(t_global, y_global[0], label="theta_e")
    plt.subplot(2,1,2)
    plt.plot(t_global, y_global[1], label="v_c1")
    plt.plot(t_global, y_global[2], label="v_c2")
    plt.plot(t_global, y_global[3], label="v_c3")
    plt.plot(t_global, y_global[4], label="v_c4")
    plt.legend()
    plt.show()
