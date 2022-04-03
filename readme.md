Following repository is a minimalistic environment for experimenting with 
Control Barrier Functions (CBF).

The classical benchmark cartpole is used as the plant model. To keep it simple
no simulator is used; instead, full non-linear dynamics of the cartpole is integrated.

Folder structure (ordered with increasing difficulty):

* `acc.py`: Most well-known, simple example: adaptive cruise control via CBF-QP. (not related to cartpole)

* `cartpole.py`: Cartpole dynamics and simulation, all controllers use it as plant
* `nominal_control.py`: Nominal controllers
* `cbf_deterministic`: Safety-Critical control with classical CBF
* `cbf_learning`: Safety-Critical control with model uncertainty, CBF + learning (TODO)
* `ecbf_deterministic`: Exponential CBF with known model (TODO)
* `ecbf_learning`: Exponential CBF with unknown model, ECBF + learning (TODO)

