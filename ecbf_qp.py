"""
This example is closely based on cbf_qp.py.

For the critical part, see _asif method of ASIF.
"""
from nominal_control import ControlLQR
from cartpole import CartPole, Controller, simulate, G

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
    def __init__(self, nominal_control, cp: CartPole, barrier_cart_pos,
            gamma_1, gamma_2, asif_enabled=True):
        """
        For our case of cartpole, limitations on cart *position* is enforced
        by this ASIF.
        """
        self.nominal_control = nominal_control
        self.cp = cp

        self.barrier_cart_pos = barrier_cart_pos
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        
        self.asif_enabled = asif_enabled
        # Based on the observations from cbf_qp.py, I removed the 
        # option for linear dynamics; use full, nonlinear dynamics.
        self._h_dot = self._h_dot_nonlinear

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
        if self.asif_enabled is False: 
            u_filtered = u_nominal  
        # if np.isclose(u_filtered, u_nominal)[0] == False:
        #     print(f"ASIF active! {u_nominal=}, {uV_filtered=}")

        return u_filtered
    
    def _asif(self, u_nominal, state):
        m_cart = self.cp.m_cart
        m_pole = self.cp.m_pole
        l = self.cp.l

        # objective function, same for all CBF-QP
        p = np.array([1.])
        q = np.array([-u_nominal])

        # constraints
        """
        In this case, we define h (see self._h) as a function of position. 
        Unlike velocity (cbf_qp.py), we do not get x_double_dot(x_ddot)
        in the _h_dot. To get the x_double_dot and control term we have to 
        differentiate one more time. In other words, our constraint (self._h) 
        is of relative degree 2. The most popular way of working with high
        relative degree constraints (<1) is to use exponential control barrier
        functions.

        We define the auxiliary functions h_e (see self._h_e). This allows us
        to get the control terms through h_e_dot (see self._h_e_dot).
        """
        # Adjust CBF constraints to match standart qp form: mainly seperate
        # control term (optimization parameter) and the other terms.

        delta = m_pole*np.sin(state[2])**2 + m_cart

        # copied from CartPole._dynamics, dxdt[1]
        x_ddot_minus_u = m_pole*l*(state[3]**2)*np.sin(state[2])/delta \
                + m_pole*G*np.sin(state[2])*np.cos(state[2])/delta \
                ## + u/delta  <--- this term goes to other side of inequality,
                ##                 fills in the variable g. If it was included
                ##                 the variable would become x_ddot = dxdt[1].
        x_dot = state[1]
        x = state[0]
        
        h_e_dot = -x_ddot_minus_u - self.gamma_1*x_dot  ## does not contain u*(control term)
        h_e = -x_dot + self.gamma_1*(self.barrier_cart_pos-x)

        h = np.array([h_e_dot + self.gamma_2*h_e])
        g = np.array([1/delta])  # control term from x_ddot (coefficient)


        u_filtered = solve_qp(p, q, g, h,
            # lb=np.array([-80.]), 
            # ub=np.array([80.]),
            solver="cvxopt")

        self._log['cbf_filtered'].append(self.cbf_cstr(state, u_filtered))
        self._log['cbf_nominal'].append(self.cbf_cstr(state, u_nominal))
        self._log['qp_g_filtered'].append(g@u_filtered)
        self._log['qp_g_nominal'].append(g@u_nominal)
        self._log['qp_h'].append(h)
        self._log['u_nom'].append(u_nominal)
        self._log['u_filtered'].append(u_filtered)

        return u_filtered

    def _h(self, state):
        return self.barrier_cart_pos - state[0]

    def _h_dot_nonlinear(self, state):  # NOTE: control signal is not included
        return -state[1]

    def _h_e(self, state):
        return self._h_dot(state) + self.gamma_1*self._h(state)

    def _h_e_dot(self, state, u):  # NOTE: control signal is included
        m_cart = self.cp.m_cart
        m_pole = self.cp.m_pole
        l = self.cp.l

        delta = m_pole*np.sin(state[2])**2 + m_cart

        # copied from CartPole._dynamics, dxdt[1]
        x_ddot = m_pole*l*(state[3]**2)*np.sin(state[2])/delta \
                + m_pole*G*np.sin(state[2])*np.cos(state[2])/delta \
                 + u/delta  # <-- unlike self._asif formulation, includes control term
        x_dot = state[1]

        return -x_ddot + self.gamma_2*x_dot

    def cbf_cstr(self, state, u):
        return self.gamma_2*self._h_e(state) + self._h_e_dot(state, u)


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
    ax2.axvline(asif.barrier_cart_pos, linestyle='--', color='red')
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
    if len(asif_log['cbf_nominal']) > 0:
        ax4 = fig.add_subplot(2, 4, 3)
        ax4.plot(t, asif_log['cbf_nominal'], label='u_nominal')
        ax4.plot(t, asif_log['cbf_filtered'], '--', label='u_filtered')
        ax4.legend()
        ax4.grid(True)
        ax4.set_title('CBF')

        ax5 = fig.add_subplot(2, 4, 4)
        ax5.plot(t, asif_log['qp_h'], label='h')
        ax5.plot(t, asif_log['qp_g_nominal'], '-.', label='u_nominal')
        ax5.plot(t, asif_log['qp_g_filtered'], '--', label='u_filtered')
        ax5.legend()
        ax5.grid(True)
        ax5.set_title('QP ineq. cstr')

        ax6 = fig.add_subplot(2, 4, 7)
        ax6.plot(t, asif_log['u_nom'], label='nominal')
        ax6.plot(t, asif_log['u_filtered'], '--', label='filtered')
        ax6.legend()
        ax6.grid(True)
        ax6.set_title('Control signal')

    if save:
        ani.save(f'{save}.mp4', fps=20)

    # plt.tight_layout()
    plt.subplots_adjust(wspace=0.5)
    plt.show()


if __name__ == "__main__":
    cp = CartPole(m_cart=5, m_pole=2, l=1.5)
    controller = ControlLQR(cp)

    """
    Similar to cbf_qp.py, barrier_cart_pos is carefully selected
    to very slightly overwrite the nominal control. Anything more aggressive
    causes the pole to drop down (I think it is the same reason with regular 
    cbf, our LQR implementation is just not robust.) 

    Anyway, feel free to try different values, the barrier does its job.
    NOTE: only positive barrier_cart_pos are implemented. For negative values,
    we could use the if statements as we used in cbf_qp.py.
    """
    asif = ASIF(controller, cp, barrier_cart_pos=1.77, gamma_1=10, gamma_2=5,
        asif_enabled=True)
    cp.set_control_law(asif)

    ## go right ~ NOTE: barrier_cart_pos is defined only for positive values.
    controller.state_ref = [1.5, 0, np.pi, 0]
    x0 = [0, 0., 178 * np.pi/180, 0]
    
    dt = 0.03
    (y, infodict), t = simulate(cp, x0=x0, dt=dt, 
        full_output=True)

    visualize(cp.l, y, t, dt, asif, infodict)
