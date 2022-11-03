# FLORIS Wake Modeling and Wind Farm Controls Software - PyTorch implementation

This codebase includes two contributions: 
1. A differentiable, learning-enabled implementation of FLORIS [1] in PyTorch (FLORIS-PT). This is a simplified version of FLORIS, which only implements the Gauss model, and is based on https://github.com/bayc/floris/blob/feature/vectorize/examples/_getting_started/floriz_ez_gauss.py. 
2. A hybrid model- and learning-based method for wind farm control implemented in PyTorch. This method is akin to Differentiable Predictive Control (DPC) [2] using a steady state model; the policy is a neural net trained by differentiating through FLORIS-PT.

New users should start with the examples.ipynb file, which shows how to use FLORIS-PT and how to train and test the DPC policy.

ACC.ipynb contains the code used to generate results for our submission to the 2023 American Control Conference [3].

[1] NREL, FLORIS v. 3.2, available at: https://github.com/NREL/floris, 2022.

[2] J. Drgoňa, A. Tuor, and D. Vrabie, Learning constrained adaptive differentiable predictive control policies with guarantees, arXiv:2004- 11184v6 [eess.SY], Jan. 2022.

[3] C. Adcock, J. King, Differentiable predictive control for adaptive wake steering, American Control Conference, 2023 (submitted.)
