Following repository is a minimalistic environment for playing out with 
Control Barrier Functions (CBF).

The classical benchmark cartpole is used as the plant model. To keep it simple
no simulator is used; instead, full dynamics of the cartpole is integrated.

Folder structure:

* `acc.py`: Most well-known, simple example: adaptive cruise control via CBF-QP


* `cartpole.py`: Cartpole dynamics and simulation
* `nominal_control.py`: Nominal controllers
* `cbf_deterministic`: Safety-Critical control with classical CBF
* `cbf_learning`: Safety-Critical control with model uncertainty, CBF + learning (TODO)
* `ecbf_deterministic`: Exponential CBF with known model (TODO)
* `ecbf_learning`: Exponential CBF with unknown model, ECBF + learning (TODO)

