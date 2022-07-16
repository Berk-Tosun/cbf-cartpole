""""""

from cartpole import Controller, CartPole, simulate, G
from nominal_control import ControlLQR

from cbf_qp import visualize

import numpy as np
from qpsolvers import solve_qp


class ASIF_ID(Controller):
    def __init__(self, nominal_control, cp: CartPole, barrier_cart_vel, gamma):
        self.nominal_control = nominal_control
        self.cp = cp

        self.barrier_cart_vel = barrier_cart_vel
        self.gamma = gamma
        
        self._log = {
            'cbf_nominal': [],
            'cbf_filtered': [],
            'qp_g_nominal': [],
            'qp_g_filtered': [],
            'qp_h': [],
            'u_nom': [],
            'u_filtered': [],
        }

    def control_law(self, state):
        u_nominal = self.nominal_control(state)
        u_filtered = self._asif(u_nominal, state)
        return u_filtered

    def _asif(self, u_nominal, state):
        # objective function
        p = np.zeros((3, 3))
        p[2, 2] = 1
        # p = np.eye(3)
        q = np.array([0, 0, -u_nominal[0]])

        # equality cstr, equations of motion
        D, H, B = self.cp.get_eom(state)
        # H = np.zeros((2, 1))  # nonlinear effects have little contribution
        a = np.concatenate((D, -B), axis=1)
        b = -H

        # inequality cstr, cbf
        g = np.array([1.0, 0, 0]).reshape(1, 3)
        h = np.array([self.gamma*(self.barrier_cart_vel - state[1])])

        u_filtered = solve_qp(p, q, g, h, a, b, solver='cvxopt')

        self._log['cbf_filtered'].append(0)
        self._log['cbf_nominal'].append(0)
        self._log['qp_g_filtered'].append(0)
        self._log['qp_g_nominal'].append(0)
        self._log['qp_h'].append(0)
        self._log['u_nom'].append(u_nominal)
        self._log['u_filtered'].append(u_filtered[2])

        return u_filtered[2]

    def _h(self, state):
        return self.barrier_cart_vel - state[1]

if __name__ == '__main__':
    cp = CartPole(m_cart=5, m_pole=2, l=1.5)
    controller = ControlLQR(cp)

    asif = ASIF_ID(controller, cp, barrier_cart_vel=0.45, gamma=10)
    cp.set_control_law(asif)

    # go right
    controller.state_ref = [1.5, 0, np.pi ,0]
    x0 = [0, 0, 178 * np.pi/180, 0]

    dt = 0.03
    (y, infodict), t = simulate(cp, x0=x0, dt=dt, 
        full_output=True)

    visualize(cp.l, y, t, dt, asif, infodict)
