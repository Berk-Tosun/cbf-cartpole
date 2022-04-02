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

          s.t.  h_dot(x, u_f) >= -gamma*h(x)
    
              ___________________           ________
         x   |                   |  u_nom  |        |   u_filtered
      -----> |  nominal control  | ------> |  ASIF  |  ------------->
             |___________________|         |________|

    """
    def __init__(self, nominal_control, cp: CartPole, barrier_cart_vel=1,
            gamma=1, asif_enabled=True):
        self.nominal_control = nominal_control
        self.cp = cp

        self.barrier_cart_vel = barrier_cart_vel
        self.gamma = gamma
        
        self.asif_enabled = asif_enabled

        self._log = {
            'cbf_nom': [],
            'qp_g': [],
            'qp_h': [],
            'u_nom': [],
            'u_filtered': [],
        }

    def control_law(self, state):
        u_nominal = self.nominal_control(state)
        u_filtered = self._asif(u_nominal, state)
        if self.asif_enabled is False:
            u_filtered = u_nominal  
        # if np.isclose(u_filtered, u_nominal)[0] == False:
        #     print(f"ASIF active! {u_nominal=}, {u_filtered=}")
        return u_filtered

    def _asif(self, u_nominal, state):
        p = np.array([1.])
        q = np.array([-u_nominal]).flatten()
        g = np.array([-1/self.cp.m_1])
        h = np.array([-G*self.cp.m_2/self.cp.m_1*state[2] 
            + self.gamma*(self.barrier_cart_vel + state[1])])

        u_filtered = solve_qp(p, q, g, h,
            # lb=np.array([-100.]), 
            # ub=np.array([100.]),
            solver="cvxopt")

        self._log['cbf_nom'].append(self._h(state))
        self._log['qp_g'].append(g@u_filtered)
        self._log['qp_h'].append(h)
        self._log['u_nom'].append(u_nominal)
        self._log['u_filtered'].append(u_filtered)

        # u_filtered = np.clip(u_filtered, -100, 100)
        return u_filtered

    def _h(self, state):
        return self.barrier_cart_vel + state[1]



def visualize(l, y, t, dt, asif: ASIF, infodict, save=None):
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
    ax = fig.add_subplot(241, autoscale_on=True, aspect='equal',\
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
    ax = fig.add_subplot(2, 4, 2)
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.grid()
    ax.plot(t, y[:, 0], color='C1',label='x')
    ax.plot(t, y[:, 1], '--', c='C1', label='x_dot')
    ax.plot(t, y[:, 2], c='C2', label='theta')
    ax.plot(t, y[:, 3], '--', c='C2', label='theta_dot')
    ax.legend()

    ax2 = fig.add_subplot(2, 4, 5)
    ax2_plt = ax2.scatter(y[:, 0], y[:, 1], c=t, alpha=0.2)
    ax2.set_title("States")
    ax2.set_xlabel("Cart position")
    ax2.set_ylabel("Cart velocity")
    ax2.grid(True)
    ax2.axhline(color='black')
    ax2.axhline(asif.barrier_cart_vel, linestyle='--', color='black')
    ax2.axhline(-asif.barrier_cart_vel, linestyle='--', color='black')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(ax2_plt, cax=cax, orientation='vertical')
    # cbar.set_label('Time')

    ax3 = fig.add_subplot(2, 4, 6)
    ax3_plt = ax3.scatter(y[:, 2], y[:, 3], c=t, alpha=0.2)
    ax3.set_title("States")
    ax3.set_xlabel("Pitch")
    ax3.set_ylabel("Pitch dot")
    ax3.grid(True)
    ax3.axhline(color='black')
    ax3.axvline(3.14, color='black')
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(ax3_plt, cax=cax, orientation='vertical')
    # cbar.set_label('Time')

    asif_log = asif.match_log_with_time(t, infodict['nfe'])
    if len(asif_log['cbf_nom']) > 0:
        ax4 = fig.add_subplot(2, 4, 3)
        ax4.plot(t, asif_log['cbf_nom'])
        ax4.axhline(color='black')
        ax4.set_title('CBF')

        ax5 = fig.add_subplot(2, 4, 4)
        ax5.plot(t, asif_log['qp_h'], label='h')
        ax5.plot(t, asif_log['qp_g'], '--', label='g')
        ax5.legend()
        ax5.set_title('QP ineq. cstr')

        ax6 = fig.add_subplot(2, 4, 7)
        ax6.plot(t, asif_log['u_nom'], label='nominal')
        ax6.plot(t, asif_log['u_filtered'], '--', label='filtered')
        ax6.legend()
        ax6.set_title('Control signal')

    if save:
        ani.save(f'{save}.mp4', fps=20)

    # plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    cp = CartPole(m_1=5, m_2=2, l=1.5)
    
    controller = ControlLQR(cp)
    # print(f'{controller.poles=}')

    asif = ASIF(controller, cp, barrier_cart_vel=0.35, asif_enabled=False)
    # asif = ASIF(controller, cp, barrier_cart_vel=0.35)
    cp.set_control_law(asif)

    controller.state_ref = [-1, 0, np.pi, 0]

    dt = 0.05
    (y, infodict), t = simulate(cp, x0=[0, 0, 183 * np.pi/180, 0], dt=dt, 
        t_end=10, full_output=True)

    visualize(cp.l, y, t, dt, asif, infodict)

