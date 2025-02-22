#%% Basic libraries for matrix manipulation and math functions
import math
import numpy as np
import torch
#from torch.autograd.functional.jacobian

#%%
# NOTE! There is a combination of numpy and torch thus if changing something use Tensors!
# torch.autograd.functional.jacobian jacobian from a function and tensor
class RobustKalman():
    def __init__(self, SysModel, test_data, c : float = 1e-3):
        self.model = SysModel # Store the system model for f and h
        self.x0 = torch.transpose(SysModel.m1x_0, 0, 1) # x0
        #self.V0 = SysModel.m2x_0 # P0
        self.V0 = 1e-3*torch.eye(2, 2)
        self.Q = SysModel.Q
        self.R = SysModel.R
        self.T = SysModel.T_test # The number of samples in the test data
        self.y = test_data
        self.c = c
        
        # Preallocation of memory
        self.n = torch.Tensor.numpy(self.Q).shape[0]
        self.p = torch.Tensor.numpy(self.R).shape[0]
        
        self.Xrekf = torch.zeros(self.n, self.T+1)
        self.Xrekf[:, 0] = self.x0
        
        self.Xn = torch.zeros(self.n, self.T)
        
        self.V = torch.zeros(self.n, self.n, self.T+1)
        self.V[:,:,0] = 1e-3*torch.eye(2, 2)
        
        self.A = torch.zeros(self.n, self.n, self.T)
        self.C = torch.zeros(self.p, self.n, self.T)
        self.G = torch.zeros(self.n, self.p, self.T)
        self.th = torch.zeros(self.T)
        
    # Numerical Jacobian Computation (This is important for us since we are using the non-linear model)
    def fnComputeJacobianF(self, x_n_temp):
        f_jac = torch.autograd.functional.jacobian(self.model.f, x_n_temp)
        return f_jac
    
    def fnComputeJacobianH(self, x_rekf_temp):
        h_jac = torch.autograd.functional.jacobian(self.model.f,x_rekf_temp)
        return h_jac
    
    def fnComputeTheta(self, P_pred):
        # NOTE: In the computation of theta there seems to be something wrong!
        
        value = torch.tensor([1])
        t1 = torch.tensor([0])
        e = torch.linalg.eig(P_pred)[0]
        r = torch.max(torch.abs(e))
        t2 = (1-1e-5)*(torch.pow(r,-1))
        
        while torch.abs(value) >= 1e-7:
            theta = 0.5*(t1+t2)
            value = torch.trace(torch.Tensor.inverse(torch.eye(self.n) - theta * P_pred)-torch.eye(self.n)) + torch.log(torch.det(torch.eye(self.n) - theta * P_pred)) - self.c
            print(value)
            if value > 0:
                t2 = theta
            else:
                t1 = theta
        
        return theta
    
    # Computation of the REKF
    def fnREKF(self):
        
        for i in range(0, self.T):
            # C_t
            self.C[:, :, i] = self.fnComputeJacobianH(self.Xrekf[:,i])
            
            # L_t
            L = self.V[:,:,i] @ torch.transpose(self.C[:,:,i], 0, 1) @ torch.Tensor.inverse(self.C[:,:,i] @ self.V[:, :, i] @ torch.transpose(self.C[:,:,i], 0, 1) + self.R)
            
            # h(\hat x_t)
            hn = self.model.h(self.Xrekf[:,i])
            
            # \hat x_t|t
            self.Xn[:, i] = self.Xrekf[:, i] + L @ (self.y[:,i] - hn)
            
            # A_t
            self.A[:, :, i] = self.fnComputeJacobianF(self.Xn[:,i])
            
            # G_t
            self.G[:, :, i] = self.A[:, :, i] @ L
            
            # \hat x_t+1
            self.Xrekf[:, i+1] = self.model.f(self.Xn[:, i])
            
            # P_t+1 - The massive fucking riccatti equation
            P = self.A[:, :, i] @ self.V[:, :, i] @ torch.transpose(self.A[:, :, i], 0, 1) - self.A[:, :, i] @ self.V[:, :, i] @ torch.transpose(self.C[:, :, i], 0, 1) @ torch.Tensor.inverse(self.C[:, :, i] @ self.V[:, :, i] @ torch.transpose(self.C[:, :, i], 0, 1)) @ self.C[:, :, i] @ self.V[:, :, i] @ torch.transpose(self.A[:, :, i], 0, 1) + self.Q

            # th_t
            self.th[i] = self.fnComputeTheta(P)
            
            # V_t+1
            self.V[:, :, i+1] = torch.Tensor.inverse(torch.Tensor.inverse(P) - self.th[i] * torch.eye(self.n))
            
        return [self.Xrekf, self.V]
    
    
    

    

    






