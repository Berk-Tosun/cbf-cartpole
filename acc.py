"""
Sanity check, mainly for solve_qp
"""
import numpy as np
import matplotlib.pyplot as plt
from qpsolvers import solve_qp

# system constants
m = 1650
g = 9.81
c_d = 0.3
v_0 = 14
v_d = 24
f_0 = 0.1
f_1 = 5
f_2 = 0.25
T_h = 1.8

# tunable parameters
alpha = 1

def F_r(v):
    return f_0 + f_1*v +f_2*v**2

def dynamics(x, u):
    """
    x = {position, velocity, distance to lead vehicle}
    """
    dxdt = np.zeros_like(x)

    dxdt[0] = x[1]
    dxdt[1] = -1/m*F_r(x[1]) + 1/m*u
    dxdt[2] = v_0 - x[1]

    return dxdt

def h(x):
    return x[2] - T_h*x[1] - 1/2 * (x[1] - v_0)**2 / (c_d*g)

def h_dot(x, u):
    return 1/m * (T_h + (x[1] - v_0) / (c_d*g)) * (F_r(x[1]) - u) + (v_0 - x[1])

def cbf_cstr(x, u):
    return h_dot(x, u) + alpha * h(x)

def simulate(dynamics, x_0, t_end, dt=0.1,):
    t = np.arange(0, t_end, dt)

    x_prev = np.array(x_0)
    y = []
    for ts in t:
        y.append(x_prev)
        # u = control_nom(x_prev)
        u = control_safe(x_prev)
        x = dynamics(x_prev, u) * dt + x_prev
        x_prev = x
    y = np.array(y)

    return y, t

def control_nom(x):
    K = 1000.0
    return K*(v_d - x[1])

cbf_t = []
h_t = []
u_nom_t = []
u_filtered_t = []
def control_safe(x):
    u_nom = control_nom(x)
    # u_filtered = u_nom

    p = np.array([1.])
    q = np.array([-u_nom])

    term = 1/m * (T_h + (x[1] - v_0) / (c_d*g))
    _g = np.array([term])
    _h = np.array([term*F_r(x[1]) + (v_0 - x[1]) + alpha*h(x)])

    u_filtered = solve_qp(p, q, _g, _h,
        solver="cvxopt")

    cbf_t.append(cbf_cstr(x, u_filtered))
    h_t.append(h(x))
    u_nom_t.append(u_nom)
    u_filtered_t.append(u_filtered)

    return u_filtered

if __name__ == "__main__":
    x_0 = [0, 20, 160]
    y, t = simulate(dynamics, x_0, 40)

    fig, axs = plt.subplots(3, 2)

    axs[0, 0].plot(t, y[:, 0])
    axs[0, 0].set_title('p')
    axs[1, 0].plot(t, y[:, 1])
    axs[1, 0].set_title('v')
    axs[2, 0].plot(t, y[:, 2])
    axs[2, 0].set_title('z')

    axs[0, 1].plot(t, cbf_t)
    axs[0, 1].set_title('cbf cstr')
    axs[1, 1].plot(t, h_t)
    axs[1, 1].set_title('cbf')
    axs[2, 1].plot(t, u_nom_t)
    axs[2, 1].plot(t, u_filtered_t, '--')
    axs[2, 1].set_title('control')
    axs[2, 1].grid()

    plt.tight_layout()
    plt.show()

