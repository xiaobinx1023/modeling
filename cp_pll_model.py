import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
# -----------------------------
# 1. Define the CP-PLL parameters
# -----------------------------
Icp   = 100e-6
Kcp = Icp / 2 / np.pi     # Charge pump current [A]
Kvdd = 20e9 * 2 * np.pi
Cp   = 65e-12     # Capacitor C1 [F]
C2   = 5e-12     # Capacitor C2 [F]
Cff = 150e-15
R1   = 20e3     # Resistor R1 [Ohms]
R2   = 1e3      # Resistor R2 [Ohms]
Kvco = 3e9 * 2 * np.pi      # VCO gain [rad/(s*V)]
A = 100 # error amp gain
wp = 10e3 * 2 * np.pi
wfr=14e9* 2 * np.pi
wref = 100e6 * 2 * np.pi      # Reference freq [rad/s]

# Initial conditions:
#   y[0] = theta_e(0)  (initial phase error)
#   y[1] = v_c1(0)
#   y[2] = v_c2(0)
y0 = [0.2, 0.0, 0.0]

# -----------------------------
# 2. Define piecewise ODE system
# -----------------------------
def cp_pll_ode(t, y):
    """
    y = [theta_e, v_c1, v_c2]
    Returns dy/dt for the simplified CP-PLL model.
    """
    theta_e, v_c1, v_c2 = y

    # Piecewise current:
    # For real designs, you might also consider dead zones or additional logic
    if theta_e > 0.0:
        ip = +Icp
    else:
        ip = -Icp

    # ODEs:
    # C1 dv_c1/dt = ip - (v_c1 / R1)
    # dv_c1_dt = (ip - (v_c1 / R1)) / Cp

    # # C2 dv_c2/dt = (v_c1 / R1) - (v_c2 / R2)
    # dv_c2_dt = ((v_c1 / R1) - (v_c2 / R2)) / C2
    # # d theta_e / dt = wref - Kvco*v_c2
    # dtheta_dt = wref - Kvco * v_c2

    dv_c1_dt = ip/C2 + v_c2/R1/C2 - v_c1/R1/C2
    dv_c2_dt = v_c1/R1/Cp - v_c2/R1/Cp
    dtheta_dt = wref - (Kvco * v_c1 + wfr)/160
    return [dtheta_dt, dv_c1_dt, dv_c2_dt]

# -----------------------------
# 3. Integrate using solve_ivp
# -----------------------------
t_span = (0, 0.000001)   # 20 ms total simulation
t_eval = np.linspace(t_span[0], t_span[1], 1000000)

solution = solve_ivp(cp_pll_ode, t_span, y0, t_eval=t_eval, rtol=1e-8, atol=1e-10)

# Extract solution
theta_sol = solution.y[0]
vc1_sol   = solution.y[1]
vc2_sol   = solution.y[2]
time      = solution.t

# -----------------------------
# 4. Plot the results
# -----------------------------
plt.figure(figsize=(10,6))

plt.subplot(3,1,1)
plt.plot(time, theta_sol, 'b-', label='Phase Error')
plt.grid(True)
plt.ylabel('theta_e [rad]')
plt.title('Piecewise CP-PLL ODE Integration')

plt.subplot(3,1,2)
plt.plot(time, vc1_sol, 'r-', label='v_c1')
plt.grid(True)
plt.ylabel('v_c1 [V]')

plt.subplot(3,1,3)
plt.plot(time, vc2_sol, 'g-', label='v_c2')
plt.grid(True)
plt.ylabel('v_c2 [V]')
plt.xlabel('Time [s]')

plt.tight_layout()
plt.show()




# Suppose we have an analytical solution vC1(t) = vC1_analytical(t)
# or we can define an ODE for vC1(t). For simplicity, let's assume an
# analytic expression. We'll just mock one here:
def vC1_analytical(t):
    # Replace with the actual expression from eq. (14)-(17)
    return np.sin(1000 * t)  # Example placeholder

def f_tp(tp):
    # Implements eq. (20), e.g.:
    # f(tp) = ∫[0 to tp] vC1(τ) dτ + ω_r * tp + θ_0 - 2π
    # We'll just do a numeric integral for demonstration:

    # numeric integration of vC1_analytical over [0, tp]
    npts = 200
    tvals = np.linspace(0, tp, npts)
    vvals = vC1_analytical(tvals)
    integral = np.trapz(vvals, tvals)

    # Then combine with the other terms from eq. (19)-(20)
    # e.g.  w_r*tp + ...
    # For demonstration:
    w_r = 2000
    theta_0 = 0
    return integral + w_r*tp + theta_0 - 2*np.pi

# Now we do a root find for tp
bracket_min, bracket_max = 0.0, 0.01  # guess a time window
sol = root_scalar(f_tp, bracket=[bracket_min, bracket_max])

if sol.converged:
    tp_solution = sol.root
    print(f"Found tp = {tp_solution:.6g}")
else:
    print("No solution found for tp in the given bracket.")
