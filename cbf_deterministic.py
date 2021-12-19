from cartpole import Controller, CartPole, simulate, G
from nominal_control import ControlLQR

import numpy as np
from qpsolvers import solve_qp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.animation as animation

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


def visualize(l, y, t, dt, asif, save=None):
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
    ax = fig.add_subplot(231, autoscale_on=True, aspect='equal',\
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
    ax = fig.add_subplot(2, 3, 2)
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.grid()
    ax.plot(t, y[:, 0])
    ax.plot(t, y[:, 2])

    ax2 = fig.add_subplot(2, 3, 4)
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

    ax3 = fig.add_subplot(2, 3, 5)
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

    ax4 = fig.add_subplot(2, 3, 3)
    ax4.plot(asif._log['cbf'])
    ax4.axhline(color='black')

    if save:
        ani.save(f'{save}.mp4', fps=20)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    cp = CartPole(m_1=5, m_2=2, l=1.5)
    
    controller = ControlLQR(cp)
    print(f'{controller.poles=}')

    asif = ASIF(controller, cp, barrier_cart_pos=0.4)
    cp.set_control_law(asif)

    controller.state_ref = [-1, 0, np.pi, 0]

    dt = 0.1
    y, t = simulate(cp, x0=[0, 0, 170 * np.pi/180, 0], dt=dt)

    visualize(cp.l, y, t, dt, asif)

