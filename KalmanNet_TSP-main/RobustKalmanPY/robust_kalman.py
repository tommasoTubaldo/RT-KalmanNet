#%% Basic libraries for matrix manipulation and math functions

import torch
import time
import torch.nn.functional as func
from KNet.RT_KalmanNet_nn import RT_KalmanNet_nn

#%%
# NOTE! There is a combination of numpy and torch thus if changing something use Tensors!
class RobustKalman():
    def __init__(self, SysModel, test_data, c : float = 1e-3, hard_coded: bool = False,use_nn: bool = False):
        
        # Select whether to use the NN or regular REKF model
        self.use_nn = use_nn
        
        # Import the model from the SysModel class
        self.model = SysModel 
        self.x0 = torch.transpose(SysModel.m1x_0, 0, 1)
        self.Q = SysModel.Q
        self.R = SysModel.R
        self.T = SysModel.T 
        self.y = test_data
        self.c = torch.tensor(c)
        self.hard_coded = hard_coded
        
        # Preallocation of memory for the computation
        self.n = torch.Tensor.numpy(self.Q).shape[0] #state dimension
        self.p = torch.Tensor.numpy(self.R).shape[0] #output dimension
        if self.use_nn:
            self.Xrekf = torch.zeros(self.n, self.T+1, requires_grad=True) 
        else:
            self.Xrekf = torch.zeros(self.n, self.T+1)  
        self.Xrekf_prev = self.x0.squeeze(0)
        self.Xn = torch.zeros(self.n, self.T)
        self.V = torch.zeros(self.n, self.n, self.T+1) 
        self.V_prev = 1e-3*torch.eye(2, 2)
        self.A = torch.zeros(self.n, self.n, self.T) 
        self.C = torch.zeros(self.p, self.n, self.T) 
        self.G = torch.zeros(self.n, self.p, self.T)
        self.th = torch.zeros(self.T) 
        
        if self.use_nn:
            self.nn = RT_KalmanNet_nn(self.p,10,[50],1)
        
    # Below one can choose to use either the closed form Jacobian or the numerical one from Pytorch
    def fnComputeJacobianF(self, x_n_temp):
        if self.hard_coded:
            f_jac = torch.tensor([[self.model.alpha*self.model.beta*(torch.cos(self.model.phi+ self.model.beta*x_n_temp[0])), 0],[0, self.model.alpha*self.model.beta*(torch.cos(self.model.phi+ self.model.beta*x_n_temp[1]))]])
        else:
            f_jac = torch.autograd.functional.jacobian(self.model.f, x_n_temp)
        return f_jac
    
    def fnComputeJacobianH(self, x_rekf_temp):
        if self.hard_coded:
            h_jac = torch.tensor([[2*self.model.a*self.model.b*(self.model.c + self.model.b*x_rekf_temp[0]), 0],[0, 2*self.model.a*self.model.b*(self.model.c + self.model.b*x_rekf_temp[1])]])
        else:
            h_jac = torch.autograd.functional.jacobian(self.model.h,x_rekf_temp)
            
        return h_jac
    
    def fnComputeTheta(self, P_pred):
        
        value = torch.tensor([1])
        t1 = torch.tensor([0])
        e = torch.linalg.eig(P_pred)[0]
        r = torch.max(torch.abs(e))
        t2 = (1-1e-5)*(torch.pow(r,-1))
        
        while torch.abs(value) >= 1e-5:
            theta = 0.5*(t1+t2)
            value = torch.trace(torch.linalg.solve(torch.eye(self.n) - theta * P_pred,torch.eye(self.n))-torch.eye(self.n)) + torch.log(torch.det(torch.eye(self.n) - theta * P_pred)) - self.c
            if value > 0:
                t2 = theta
            else:
                t1 = theta
        
        return theta

    # Computation of the REKF
    def fnREKF(self, train: bool = False):
        # Setting the NN to training or evaluation
        if self.use_nn:
            if train:
                self.nn.train()
            else:
                self.nn.eval()
                torch.no_grad()

        start = time.time()

        # Forward Step
        for i in range(0, self.T):

            # C_t
            self.C[:, :, i] = self.fnComputeJacobianH(self.Xrekf_prev)

            # L_t
            L = self.V_prev @ torch.transpose(self.C[:, :, i], 0, 1) @ torch.linalg.solve(
                self.C[:, :, i] @ self.V_prev @ torch.transpose(self.C[:, :, i], 0, 1) + self.R, torch.eye(self.p))

            # h(\hat x_t)
            hn = self.model.h(self.Xrekf_prev)

            # \hat x_t|t
            self.Xn[:, i] = self.Xrekf_prev + (L @ (self.y[:, i] - hn))

            # A_t
            self.A[:, :, i] = self.fnComputeJacobianF(self.Xn[:, i])

            # G_t
            self.G[:, :, i] = self.A[:, :, i] @ L

            # \hat x_t+1
            self.Xrekf = self.Xrekf.clone()
            self.Xrekf[:, i + 1] = torch.squeeze(self.model.f(self.Xn[:, i]))
            self.Xrekf_prev = self.Xrekf[:, i + 1]

            # P_t+1
            P = self.A[:, :, i] @ self.V_prev @ torch.transpose(self.A[:, :, i], 0, 1) - self.A[:, :,i] @ self.V_prev @ torch.transpose(self.C[:, :, i], 0, 1) @ torch.linalg.solve(self.C[:, :, i] @ self.V_prev @ torch.transpose(self.C[:, :, i], 0, 1) + self.R,torch.eye(self.p)) @ self.C[:, :, i] @ self.V_prev @ torch.transpose(self.A[:, :, i], 0, 1) + self.Q

            if self.use_nn:
                # Compute the Innovation Difference
                delta_y = self.y[:, i] - hn

                # Forward Step
                self.c = self.nn(delta_y)

            # th_t
            self.th[i] = self.fnComputeTheta(P)

            # V_t+1
            self.V[:, :, i + 1] = torch.linalg.solve(
                torch.linalg.solve(P, torch.eye(self.n)) - self.th[i] * torch.eye(self.n), torch.eye(self.n))
            self.V_prev = self.V[:, :, i + 1]

        end = time.time()
        t = end - start

        return [self.Xrekf, self.V, t]