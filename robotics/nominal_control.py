"""
From ilqr/examples/cartpole.ipynb
"""

import numpy as np
import matplotlib.pyplot as plt

from ilqr import iLQR
from ilqr.cost import QRCost
from ilqr.dynamics import constrain
from ilqr.examples.cartpole import CartpoleDynamics

def on_iteration(iteration_count, xs, us, J_opt, accepted, converged):
    J_hist.append(J_opt)
    info = "converged" if converged else ("accepted" if accepted else "failed")
    final_state = dynamics.reduce_state(xs[-1])
    print("iteration", iteration_count, info, J_opt, final_state)

dt = 0.05
pole_length = 1.0
m_cart = 1.0
m_pole = 0.1
dynamics = CartpoleDynamics(dt, l=pole_length)