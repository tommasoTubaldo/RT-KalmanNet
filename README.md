Final Project of the course "Learning Dynamical Systems" regarding the implementation and evaluation of a Robust Tailored KalmanNet.

# Abstract

The well-known SOTA model for non-linear state estimation is the Extended Kalman Filter[1-3]. However, if the knowledge of the noise covariances is not available or there are model uncertanties, the EKF performs poorly due to lack of robustness. In order to improve the robustness of the filter, the Robust EKF[4] has been formulated. The REKF is based on the formulation of an ambiguity set, in which the actual model $\tilde{M_t}$ is assumed to be inside the ball with center the nominal model $M_T$ and radius the tolerance $c_t$ in terms of the KL divergence.
In addition, in the past few years data-driven formulations of the Kalman filter have solved the issue related to the knowledge of the noise covariances by adopting neural networks for the estimantion of the Kalman gain. One particular example is KalmanNet[5], where an interpretable Recurrent Neural Networl (RNN) is used. Still, these NN-based models are subjected to model uncertanties.
The objective of the project is to extend the capability of the REKF, in particular by adopting a NN module for the estimation of the tolerance $c_t$.

References
[1] K. Reif, S. Gunther, E. Yaz, and R. Unbehauen, “Stochastic stability of the discrete-time extended Kalman filter,” IEEE Transactions on
Automatic control, vol. 44, no. 4, pp. 714–728, 1999.
[2] X. Wang and E. E. Yaz, “Second-order fault tolerant extended Kalman filter for discrete time nonlinear systems,” IEEE Transactions on
Automatic Control, vol. 64, no. 12, pp. 5086–5093, 2019.
[3] A. Barrau and S. Bonnabel, “The invariant extended Kalman filter as a stable observer,” IEEE Transactions on Automatic Control, vol. 62,
no. 4, pp. 1797–1812, 2016.
[4] A. Longhini, M. Perbellini, S. Gottardi, S. Yi, H. Liu and M. Zorzi, "Learning the tuned liquid damper dynamics by means of a robust EKF," 
2021 American Control Conference (ACC), New Orleans, LA, USA, 2021, pp. 60-65, doi: 10.23919/ACC50511.2021.9483172.
[5] G. Revach, N. Shlezinger, X. Ni, A. L. Escoriza, R. J. G. van Sloun and Y. C. Eldar, "KalmanNet: Neural Network Aided Kalman Filtering for 
Partially Known Dynamics," in IEEE Transactions on Signal Processing, vol. 70, pp. 1532-1547, 2022, doi: 10.1109/TSP.2022.3158588.
