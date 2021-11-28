import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.animation as animation
import numpy as np


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
    cbar.set_label('Time')

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
    cbar = fig.colorbar(ax2_plt, cax=cax, orientation='vertical')
    cbar.set_label('Time')

    if save:
        ani.save(f'{save}.mp4', fps=20)

    plt.tight_layout()
    plt.show()

