CBF-ID-QP
---------

Inverse Dynamics formulation of CBP. Motivated by Reher, Jenna, Claudia Kann, and Aaron D. Ames. "An inverse dynamics approach to control lyapunov functions." 2020 American Control Conference (ACC). IEEE, 2020.

More advanced and useful then the regular CBF-QP formulation, for robotics.
Writing dynamics in first order requires inversion of mass matrix. Additionally,
for systems with contact, contact constraints are enforced directly with inverse
dynamics formulation. In addition, structure and sparsity of the equations of 
motion could be exploited to come up with very fast controller.

ID-QP formulations work on instantenous control signals. To get the most out of
them, planning is done by some planner/trajectory optimizer. The plan is
executed by ID-QP.

For our case, we have a simpler scheme consisting of following components:

    1. `nominal_control.py` ~ Trajectory optimization
        Inputs: initial state, target state
        Output: trajectory (evolution of states, required control inputs for 
            over a finite time interval)
    2. `cbf_id_qp.py` ~ Safety filter
        Inputs: instantenous control input (from trajectory), current state
        Output: safe instantenous control input

Additional Requirements
-----------------------

To compute the terms in the equation of motion (mass matrix, jacobians, etc.)

    Install pinocchio with 'pip install pinocchio'

To achive trajectory optimization

        1. Copy ilqr from git with 'git submodule update --init --recursive'
           (or 'git clone https://github.com/anassinator/ilqr.git')

        2. Install ilqr with 'cd ilqr && pip install -e .'

