import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import integrate
import matplotlib.animation as animation
import control

# inspired from:
# source: https://github.com/YifanJiangPolyU/cart-pole-nonlinear-control/blob/master/cart-pole-balancing.py
# Cart-pole dynamic system:
# 
#                                 O ball: m2
#                                /
#                               /
#                              /  pole: l
#                             /
#                      _____ /_____
#                     |            |   
#  Force: u -->       |            | Cart: m1
#                     |____________|
#    |---> q1 (x)        O  ︙   O
#                 ‾‾/‾‾‾/‾‾‾︙‾‾‾‾/‾‾‾/‾‾‾/‾‾  
#                           ︙    
#                           ︙ ⤻ 
#                           ︙ q2 (theta)
#                           ︙    

#####################################################
## Define dynamics - simple pendulum

# Use langrange or anything or just the source below to get equations of motion
# https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-832-underactuated-robotics-spring-2009/readings/MIT6_832s09_read_ch03.pdf
# equations 3.16 & 3.17

g = 9.8  # m/s**2

m_1 = 4.0  # kg
m_2 = 1  # kg
l = 1.25  # 85.4 / 2 * 0.01  # m
# wheel_dia = 0.39 # m

# For simulation purposes, express full (nonlinear) dynamics:
def dynamics(x, t):
    """ x: {x, x_dot, theta, theta_dot} """
    dxdt = np.zeros_like(x)

    dxdt[0] = x[1]
    dxdt[2] = x[3]

    ### with controller
    u = control_law(x)
    ### without controller
    # u = 0

    delta = m_2*np.sin(x[2])**2 + m_1

    dxdt[1] = m_2*l*(x[3]**2)*np.sin(x[2])/delta \
        + m_2*g*np.sin(x[2])*np.cos(x[2])/delta \
        + u/delta

    dxdt[3] = - m_2*(x[3]**2)*np.cos(x[2])*np.sin(x[2])/delta \
        - (m_1 + m_2)*g*np.sin(x[2])/delta/l \
        - u*np.cos(x[2])/delta/l

    return dxdt


# For control purposes, express simplified dynamics:
# Writing in the simplest form, replace pendulum with simple pendulum
# and linearize; taken from Ogata's book (Modern Control - modelling chapter).
# Linearization for pendulum up state, x = [?, ?, np.pi, ?]

## Ogata
def get_ss_A():
    return [
        [0, 1, 0, 0],
        [0, 0, m_2 * -g /m_1, 0],
        [0, 0, 0, 1],
        [0, 0, -(m_1 + m_2) * -g / (m_1*l), 0]
    ]

def get_ss_B():
    return np.array([0, 1/m_1, 0, 1/(m_1 * l)])

## Steve Brunton ~ adds damping on wheels
# def get_ss_A(m1, m2, l, d=1, pendulum_up=True, g=9.807):
#     b = 1 if pendulum_up else -1
#     return [
#         [0, 1, 0, 0],
#         [0, -d/m1, b*m2*-g/m1, 0],
#         [0, 0, 0, 1],
#         [0, -b*d/(m1*l), b*(m1+m2)*g/(m1*l), 0]
#     ]

# def get_ss_B(m1,l, pendulum_up=True):
#     b = 1 if pendulum_up else -1
#     return [0, 1/m1, 0, b/(m1 * l)]

A = get_ss_A()
B = get_ss_B()
C = [
    [1, 0, 0, 0],
    [0, 0, 1, 0]
]

model_plant = control.ss(A, B, C, 0)

# add controller

Q = np.zeros((4, 4))
np.fill_diagonal(Q, [10, 1, 10, 100])
R = 1

K_lqr, _, poles = control.lqr(model_plant, Q, R)
print(f"{poles=}")
state_ref = np.array([-1, 0., np.pi, 0.])

def control_law(x):
    return K_lqr @ (state_ref - x)

#####################################################
## Simulate

dt = 0.05
t = np.arange(0.0, 20, dt)

x = 1.0
xdot = 0.0
theta = 170 / 180*np.pi 
w = 0.0

# initial state
x0 = np.array([x, xdot, theta, w])

y = integrate.odeint(dynamics, x0, t)

#####################################################
## Animate

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

#ani.save('cart-pole-LQR.mp4', fps=20)

plt.tight_layout()
plt.show()

