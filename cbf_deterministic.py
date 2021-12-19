from matplotlib.pyplot import bar
from cartpole import Controller, CartPole, simulate
from visual import visualize
from nominal_control import ControlLQR

import numpy as np
from qpsolvers import solve_qp


class ASIF(Controller):
    """Active Set Invariance Filter
    
    Implementation of the popular CBF-QP
    
      ASIF takes in the nominal control signal(u_nom) and filters it to
      generate u_filtered. Under the hood it is an optimization problem
      with following objective function:

          u_f = argmin || u_f - u_nom ||^2

          s.t.  h_dot(x, u) >= -gamma(h, x)
    
              ___________________           ________
         x   |                   |  u_nom  |        |   u_filtered
      -----> |  nominal control  | ------> |  ASIF  |  ------------->
             |___________________|         |________|

    """
    # def __init__(self, nominal_control, theta_barrier=165):
        # self.theta_barrier = theta_barrier
    def __init__(self, nominal_control, cp: CartPole, barrier_cart_pos=1):
        self.nominal_control = nominal_control
        self.cp = cp
        self.barrier_cart_pos = barrier_cart_pos

        self._log = {
            'cbf': [],
        }

    def control_law(self, state):
        u_nominal = self.nominal_control(state)
        # u_filtered = u_nominal
        u_filtered = self._asif(u_nominal, state)
        if np.isclose(u_filtered, u_nominal)[0] == False:
            print(f"ASIF active! {u_nominal=}, {u_filtered=}")
        return u_filtered

    def _asif(self, u_nominal, state):
        gamma = 1e-3

        p = np.array([1.])
        q = np.array([-u_nominal]).flatten()
        g = np.array([-1/self.cp.m_1])
        h = np.array([G*self.cp.m_2/self.cp.m_1*state[2] 
            - gamma*state[1] + gamma])

        self._log['cbf'].append(self._h(state))

        return solve_qp(p, q, g, h,
            lb=np.array([-100.]), 
            ub=np.array([100.]),
            solver="cvxopt")

    # def _h(self, state):
    #     return abs(state[2]) - self.theta_barrier * np.pi/180

    def _h(self, state):
        return state[0] - self.barrier_cart_pos

    def _h_dot(self, state):
        return state[1]

if __name__ == "__main__":
    cp = CartPole(m_1=5, m_2=2, l=1.5)
    
    controller = ControlLQR(cp)
    print(f'{controller.poles=}')

    asif = ASIF(controller)
    cp.set_control_law(asif)

    controller.state_ref = [-1, 0, np.pi, 0]

    dt = 0.1
    y, t = simulate(cp, x0=[0, 0, 170 * np.pi/180, 0], dt=dt)

    visualize(cp.l, y, t, dt)

