from typing import Callable

import numpy as np
from scipy import integrate
import control
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.animation as animation

# inspired from:
# source: https://github.com/YifanJiangPolyU/cart-pole-nonlinear-control/blob/master/cart-pole-balancing.py

G = 9.81  # m/s**2

class Controller:
    def __init__(self):
        self._log = {}

    def __call__(self, state):
        return self.control_law(state)

    def control_law(self, state):
        raise NotImplementedError()

    def match_log_with_time(self, t, nfe):
        """
        Since scipy.integrate may make multiple function calls for each step
        the internal log will have more elements than time vector, t. Use 
        infodict['nfe'] - cumulative number of function evaluation for each 
        time step - to match log with t.

        Args: 
            t: t as returned by scipy.integrate
            nfe: infodict['nfe'] as returned by scipy.integrate

        Returns:
            log: log with same number of elements as t
        """
        nfe = np.insert(nfe, 0, 1)
        nfe = nfe - 1

        log = {}
        for k, v in self._log.items():
            try:
                log[k] = np.array(v)[nfe]
            except IndexError:
                log[k] = []

        return log


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

    def set_control_law(self, f: Controller):
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


def visualize(l, y, t, dt, save=None):
    """
    Args:
        l: Pendulum length, CartPole.l
        y: simulation outputs
        t: simulation timesteps
        dt
    """
    a_x1 = y[:, 0]
    a_y1 = 0.0

    a_x2 = l * np.sin(y[:, 2]) + a_x1
    a_y2 = -l * np.cos(y[:, 2]) + a_y1

    fig = plt.figure()
    ax = fig.add_subplot(221, autoscale_on=True, aspect='equal',\
                            xlim=(-3, 3), ylim=(-3, 3))

    ax.grid()

    line, = ax.plot([], [], 'o-', lw=2)
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    def animate(i):
        thisx = [a_x1[i], a_x2[i]]
        thisy = [a_y1, a_y2[i]]

        line.set_data(thisx, thisy)
        time_text.set_text(time_template%(i*dt))
        return line, time_text

    ani = animation.FuncAnimation(fig, animate, np.arange(1, len(y)),
        interval=30, blit=True, init_func=init)

    # time domain plot
    ax = fig.add_subplot(2, 2, 2)
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.grid()
    ax.plot(t, y[:, 0])
    ax.plot(t, y[:, 2])

    ax2 = fig.add_subplot(2, 2, 3)
    ax2_plt = ax2.scatter(y[:, 0], y[:, 1], c=t, alpha=0.2)
    ax2.set_title("States (Phase plane)")
    ax2.set_xlabel("Cart position")
    ax2.set_ylabel("Cart velocity")
    ax2.grid(True)
    ax2.axhline(color='black')
    ax2.axvline(3.14, color='black')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(ax2_plt, cax=cax, orientation='vertical')
    # cbar.set_label('Time')

    ax3 = fig.add_subplot(2, 2, 4)
    ax3_plt = ax3.scatter(y[:, 2], y[:, 3], c=t, alpha=0.2)
    ax3.set_title("States (Phase plane)")
    ax3.set_xlabel("Pitch")
    ax3.set_ylabel("Pitch dot")
    ax3.grid(True)
    ax3.axhline(color='black')
    ax3.axvline(3.14, color='black')
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(ax3_plt, cax=cax, orientation='vertical')
    # cbar.set_label('Time')

    if save:
        ani.save(f'{save}.mp4', fps=20)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # No control, only initial conditions
    cp = CartPole(m_1=5, m_2=2, l=1.5)

    dt = 0.1
    y, t = simulate(cp, x0=[0, 0, 170 * np.pi/180, 0], dt=dt)

    visualize(cp.l, y, t, dt)

