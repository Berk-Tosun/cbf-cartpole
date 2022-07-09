from cartpole import Controller, CartPole, simulate, visualize

import numpy as np
import control


class ControlLQR(Controller):
    def __init__(self, cp: CartPole,
            Q= [[10, 0,  0,   0],
                [0,  1,  0,   0],
                [0,  0, 10,   0],
                [0,  0,  0, 100]],
            R=1,
            state_ref=np.array([0, 0., np.pi, 0.])):
        self.cp = cp
        self.Q = Q
        self.R = R
        self.state_ref = state_ref

        self._gen_control()

    def _gen_control(self):
        self.K_lqr, _, self.poles = control.lqr(self.cp.get_openloop(),
            self.Q, self.R)

    def control_law(self, state):
        return self.K_lqr @ (self.state_ref - state)

    def set_reference(self, reference):
        self.state_ref = reference

if __name__ == "__main__":
    cp = CartPole(m_cart=5, m_pole=2, l=1.5)
    
    # q = np.zeros((4, 4))
    # np.fill_diagonal(q, [25, 1, 10, 100])
    controller = ControlLQR(cp, R=1) # Q=q)
    print(f'LQR controller poles: {controller.poles}')
    cp.set_control_law(controller)

    controller.state_ref = [-1, 0, np.pi, 0]

    dt = 0.1
    y, t = simulate(cp, x0=[0, 0, 170 * np.pi/180, 0], dt=dt)

    visualize(cp.l, y, t, dt)