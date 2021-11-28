from typing import Callable

import numpy as np
from scipy import integrate
import control

from visual import visualize

# inspired from:
# source: https://github.com/YifanJiangPolyU/cart-pole-nonlinear-control/blob/master/cart-pole-balancing.py

G = 9.8  # m/s**2

class CartPole:
    def __init__(self, m_1=4, m_2=1, l=1.25):
        """
        Parameters:
            m_1: Mass of Cart [kg]
            m_2: Mass at the end of pendulum [kg]
            l: Length of pendulum [m]
        
        .. todo::
            d: Viscous damping between wheels and ground [Ns/m]
        
        Cart-pole dynamic system:

            State, shown with x = [q1, q1_dot, q2, q2_dot]
            (x is also used to represent q1)
        
                                         O ball: m2
                                        /
                                       /
                                      /  pole: l
                                     /
                              _____ /_____
                             |            |   
                Force: u --> |            | Cart: m1
                             |____________|
            |---> q1 (x)        O  ︙   O
                         ‾‾/‾‾‾/‾‾‾︙‾‾‾‾/‾‾‾/‾‾‾/‾‾  
                                   ︙    
                                   ︙ ⤻ 
                                   ︙ q2 (theta)
                                   ︙    
        """
        self.m_1 = m_1
        self.m_2 = m_2
        self.l = l
        # self.d = d

        self._gen_dynamics()

    def _gen_dynamics(self):
        """
        Create / Update self.dynamics()

        For simulation purposes, express full (nonlinear) dynamics.
        x represents the state
            x: {x, x_dot, theta, theta_dot} 

        Use langrange or anything or just the source below to get equations of motion

        https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-832-underactuated-robotics-spring-2009/readings/MIT6_832s09_read_ch03.pdf

            Equations 3.16 & 3.17
        """
        m_1, m_2, l = self.m_1, self.m_2, self.l

        def _dynamics(x, t):    
            dxdt = np.zeros_like(x)

            dxdt[0] = x[1]
            dxdt[2] = x[3]

            ### with controller
            u = self.control_law(x)

            delta = m_2*np.sin(x[2])**2 + m_1

            dxdt[1] = m_2*l*(x[3]**2)*np.sin(x[2])/delta \
                + m_2*G*np.sin(x[2])*np.cos(x[2])/delta \
                + u/delta

            dxdt[3] = - m_2*(x[3]**2)*np.cos(x[2])*np.sin(x[2])/delta \
                - (m_1 + m_2)*G*np.sin(x[2])/delta/l \
                - u*np.cos(x[2])/delta/l

            return dxdt
        
        self.dynamics = _dynamics

    @staticmethod
    def dynamics(x, t):
        return RuntimeError()

    def get_ss_A(self):
        """
        For control purposes, express simplified dynamics:
        Writing equations of motion in the simplest form, 
        linearize; taken from Ogata's book (Modern Control - modelling chapter).
        Linearization for pendulum for up state; i.e., x = [?, ?, np.pi, ?]
        """
        m_1, m_2, l = self.m_1, self.m_2, self.l

        return [
            [0, 1, 0, 0],
            [0, 0, m_2 * -G /m_1, 0],
            [0, 0, 0, 1],
            [0, 0, -(m_1 + m_2) * -G / (m_1*l), 0]
        ]
        ## Steve Brunton ~ adds damping on wheels
        #     b = 1 if pendulum_up else -1
        #     return [
        #         [0, 1, 0, 0],
        #         [0, -d/m_1, b*m_2*-G/m_1, 0],
        #         [0, 0, 0, 1],
        #         [0, -b*d/(m_1*l), b*(m_1+m_2)*G/(m_1*l), 0]
        #     ]

    def get_ss_B(self):
        return np.array([0, 1/self.m_1, 0, 1/(self.m_1 * self.l)])

    def get_openloop(self, 
            C = [[1, 0, 0, 0],
                 [0, 0, 1, 0]]
            ):
        A = self.get_ss_A()
        B = self.get_ss_B()
        
        return control.ss(A, B, C, 0)

    @staticmethod
    def control_law(x):
        return 0

    def set_control_law(self, f: Callable):
        self.control_law = f  # type: ignore
        self._gen_dynamics()

    def reset_control_law(self):
        self.control_law = lambda x: 0
        self._gen_dynamics()

        
def simulate(cp: CartPole,
        x0=[0, 0, 175 * 180/np.pi, 0],
        t_end=20,
        dt=0.1, **kwargs):
    """
    Args:
        dynamics: derivatives of the states in function form
        x0: Initial state
        t_end: Simulation duration [s]
        dt: Timestep size [s]

    Returns:
        y: simulation output
        t: simulation timesteps
    """
    t = np.arange(0.0, t_end, dt)

    y = integrate.odeint(cp.dynamics, x0, t, **kwargs)
    return y, t


if __name__ == "__main__":
    # No control, only initial conditions
    cp = CartPole(m_1=5, m_2=2, l=1.5)

    dt = 0.1
    y, t = simulate(cp, x0=[0, 0, 170 * np.pi/180, 0], dt=dt)

    visualize(cp.l, y, t, dt)

