Following repository is a minimalistic environment for experimenting with 
Control Barrier Functions (CBF).

The classical benchmark cartpole is used as the plant model. To keep it simple
no simulator is used; instead, full non-linear dynamics of the cartpole is integrated.

Folder structure (ordered with increasing difficulty):

* `acc.py`: Most well-known, simple example: adaptive cruise control via CBF-QP. (not related to cartpole)

* `cartpole.py`: Cartpole dynamics and simulation, all controllers use it as plant
* `nominal_control.py`: Nominal controllers
* `cbf_deterministic`: Safety-Critical control with classical CBF

|  CBF Enabled             |  CBF Disabled            |
|--------------------------|--------------------------|
| ![CBF Enabled](https://github.com/Berk-Tosun/cbf-cartpole/blob/master/doc/cbf_cartpole_on.gif) | ![CBF Disabled](https://github.com/Berk-Tosun/cbf-cartpole/blob/master/doc/cbf_cartpole_off.gif) |

* `cbf_learning`: Safety-Critical control with model uncertainty, CBF + learning (TODO)
* `ecbf_deterministic`: Exponential CBF with known model (TODO)
* `ecbf_learning`: Exponential CBF with unknown model, ECBF + learning (TODO)

---

## What is CBF

A good introduction: https://www.youtube.com/watch?v=_Tkn_Hzo4AA&list=PLgDu6Qp_poydAuNfVJrTD53dpnVit_Pt5&index=10

Matching repository of the above presentation: https://github.com/HybridRobotics/CBF-CLF-Helper where you can get the slides.

Another introduction: https://www.youtube.com/watch?v=vmRl8swiEyc&list=PLgDu6Qp_poydAuNfVJrTD53dpnVit_Pt5&index=23
