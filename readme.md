## Introduction

Following repository is a minimalistic environment for experimenting with 
Control Barrier Functions (CBF).

The classical benchmark cartpole is used as the plant model. To keep it simple
no simulator is used; instead, full non-linear dynamics of the cartpole is integrated.

Folder structure (ordered with increasing difficulty):

* `acc.py`: Most well-known, simple example: adaptive cruise control via CBF-QP. (not related to cartpole)

* `cartpole.py`: Cartpole dynamics and simulation, all controllers use it as plant
* `nominal_control.py`: Nominal controllers
* `cbf_qp.py`: Safety-Critical control with classical CBF, constrain cart velocity

|  CBF Enabled             |  CBF Disabled            |
|--------------------------|--------------------------|
| ![CBF Enabled](https://github.com/Berk-Tosun/cbf-cartpole/blob/master/doc/cbf_cartpole_on.gif) | ![CBF Disabled](https://github.com/Berk-Tosun/cbf-cartpole/blob/master/doc/cbf_cartpole_off.gif) |

* `ecbf_qp.py`: Exponential CBF with known model, constrain cart position
* `cbf_qp_learning`: Safety-Critical control with model uncertainty, CBF + learning (TODO)
* `ecbf_qp_learning`: Exponential CBF with unknown model, ECBF + learning (TODO)
* `cbf_qp_id.py`: CBF-QP embedded into inverse dynamics formulation.

## Installaton

1. Clone the repository

    > git clone https://github.com/Berk-Tosun/cbf-cartpole
  
2. Install required packages

    > pip install -r requirements.txt

## What is CBF?

* Recommended introductory [presentation](https://youtu.be/_Tkn_Hzo4AA): Jason Choi, UC Berkeley. Matching [repository](https://github.com/HybridRobotics/CBF-CLF-Helper) of the talk, where you can get the slides and the mentioned matlab code.

* [Presentation](https://youtu.be/ZC3T_P_8xpE) by the main author of the CBF framework, Prof. Aaron Ames.

* Another introductory [presentation](https://youtu.be/vmRl8swiEyc): Air Lab Summer School 2020.

* Sample hardware application: [Multi-Layered Safety for Legged Robots via Control Barrier Functions and Model Predictive Control](http://dx.doi.org/10.13140/RG.2.2.17776.89605). Its [presentation](https://youtu.be/xZqapQU2k84) from ICRA 2021.
