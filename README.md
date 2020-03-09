Calculating errors for derivatives of benchmark functions
=====================================================

Introduction
============

In this project gradients and hessians of Rastrigin, Levy and Ackley functions are calculated (you can find the functions' setups here https://www.sfu.ca/~ssurjano/optimization.html). Using Estimagic package from python the numerical derivatives are calculated for the same functions. The relative error which is the difference between the numerical and analytical derivativs, devided by the analytical derivative,is calculated and ploted.
The project works with the waf environment <https://github.com/hmgaudecker/econ-project-templates/>`_.
The project works as follows
1. Calculation of Benchmark functions, their gradients and hessians, stored in 'model_code' folder.
2. Unit tests for Benchmark functions, their gradients and hessians, stored in 'analysis' folder.
3. Calculations and plots of relative errors for the functions and their derivatives, stored in 'analysis' folder. Plots are saved in   
   'out' folder.
