"""
See explanation below in the __name__ guard.

For critical part, see _asif method of ASIF.
"""

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
    def __init__(self, nominal_control, cp: CartPole, barrier_cart_vel,
            gamma, asif_enabled=True, use_nonlinear_dynamics=True):
        """
        For our case of cartpole, limitations on cart *velocity* is enforced
        by this ASIF.
        """
        self.nominal_control = nominal_control
        self.cp = cp

        self.barrier_cart_vel = barrier_cart_vel
        self.gamma = gamma
        
        self.asif_enabled = asif_enabled
        self.use_nonlinear_dynamics = use_nonlinear_dynamics
        if self.use_nonlinear_dynamics:
            self._h_dot = self._h_dot_nonlinear
        else:
            self._h_dot = self._h_dot_linear

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
        #     print(f"ASIF active! {u_nominal=}, {u_filtered=}")

        return u_filtered

    def _asif(self, u_nominal, state):
        m_cart = self.cp.m_cart
        m_pole = self.cp.m_pole
        l = self.cp.l

        # objective function, same for all CBF-QP
        p = np.array([1.])
        q = np.array([-u_nominal])

        # constraints
        if self.use_nonlinear_dynamics:
            # the terms come from self.h_dot_nonlinear, organized for standart 
            # qp solver format
            delta = m_pole*np.sin(state[2])**2 + m_cart
            if state[1] >= 0:  
                g = np.array([1/delta])
                h = -1 * np.array([m_pole*l*(state[3]**2)*np.sin(state[2])/delta \
                        + m_pole*G*np.sin(state[2])*np.cos(state[2])/delta]) \
                        + self.gamma*(self._h(state))
            else:
                g = np.array([-1/delta])
                h = np.array([m_pole*l*(state[3]**2)*np.sin(state[2])/delta \
                        + m_pole*G*np.sin(state[2])*np.cos(state[2])/delta]) \
                        + self.gamma*(self._h(state))
        else:
            # the terms come from self.h_dot_linear, organized for standart
            # qp solver format
            if state[1] >= 0:
                g = np.array([1/m_cart])
                h = np.array([m_pole*G/m_cart*state[2] + self.gamma*self._h(state)])
            else:
                g = np.array([-1/m_cart])
                h = np.array([-m_pole*G/m_cart*state[2] + self.gamma*self._h(state)])

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
        if state[1] >= 0:
            return self.barrier_cart_vel - state[1]
        else:
            return self.barrier_cart_vel + state[1]

    def _h_dot_nonlinear(self, state, u):
        """ Equations from cartpole._gen_dynamics._dynamics"""
        m_cart = self.cp.m_cart
        m_pole = self.cp.m_pole
        l = self.cp.l
        delta = m_pole*np.sin(state[2])**2 + m_cart

        if state[1] >= 0:
            return -1 * (m_pole*l*(state[3]**2)*np.sin(state[2])/delta \
                + m_pole*G*np.sin(state[2])*np.cos(state[2])/delta) - u/delta
        else:
            return (m_pole*l*(state[3]**2)*np.sin(state[2])/delta \
                + m_pole*G*np.sin(state[2])*np.cos(state[2])/delta) + u/delta

    def _h_dot_linear(self, state, u):
        """ Equations from cartpole.get_ss_A"""
        m_cart = self.cp.m_cart
        m_pole = self.cp.m_pole

        if state[1] >= 0:
            return m_pole*G/m_cart*state[2] - u/m_cart
        else:
            return -m_pole*G/m_cart*state[2] + u/m_cart

    def cbf_cstr(self, state, u):
        return self.gamma*self._h(state) + self._h_dot(state, u)

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
    ax2.axhline(asif.barrier_cart_vel, linestyle='--', color='red')
    ax2.axhline(-asif.barrier_cart_vel, linestyle='--', color='red')
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
    We are limiting nominal control of LQR with respect to cart velocity with 
    the use of ASIF.

    Advice: switch asif_enabled below and see its effects

    In this example, I have adjusted the parameters rather carefully; the ASIF 
    does not cause a major difference in the nominal control. If you make the
    constraint more strict, it will cause problems. See note 1 below.

    Note 1:
    Current ASIF formulation do not work well with the nominal control,
    the simple constraint on only cart velocity causes the ASIF
    to ignore pendulum state, if we deviate considerably from nominal control
    the pendulum is likely to fall. Once it has fallen, LQR cant recover since
    we are out of its operation range (for LQR we are using linearized dynamics
    around the pendulum up orientation).

    To apply it on an actual case, we better make it more robust. My guess is
    a more involved CBF, i.e. h, would do the job. An alternative is to use 
    a controller which can operate independent of the pendulum orientation.

    Note 2:
    I am surprised by the effect of using linearized dynamics on ASIF, it causes
    a much stronger correction which kicks the system out of its pendulum up
    orientation. Then, it becomes unstable as explained above.

    I have made various attempts; however, I could not get ASIF to 
    operate with linearized dynamics.
    """
    asif = ASIF(controller, cp, barrier_cart_vel=0.45, gamma=10,
        asif_enabled=True, use_nonlinear_dynamics=True)
    cp.set_control_law(asif)

    ## go right
    ## Our formulation of ASIF is quite sensitive to initial conditions,
    ## see 'Note 1' above
    controller.state_ref = [1.5, 0, np.pi, 0]
    x0 = [0, 0., 178 * np.pi/180, 0]

    ## go left
    ## Our formulation of ASIF is quite sensitive to initial conditions,
    ## see 'Note 1' above
    # controller.state_ref = [-1.5, 0, np.pi, 0]
    # x0 = [0, 0., 182 * np.pi/180, 0]
    
    dt = 0.03
    (y, infodict), t = simulate(cp, x0=x0, dt=dt, 
        full_output=True)

    visualize(cp.l, y, t, dt, asif, infodict)

