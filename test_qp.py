import numpy as np
from qpsolvers import solve_qp

P = np.array([1.])
q = np.array([0.])
G = np.array([1.])
h = np.array([-3.])


x = solve_qp(P, q, G, h, solver='cvxopt')
print(f"QP solution: x = {x}")