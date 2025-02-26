#%% Basic libraries for matrix manipulation and math functions


import math
import numpy as np
import torch
from KNet.RT_KalmanNet_nn import RT_KalmanNet_nn

#%%


# NOTE! There is a combination of numpy and torch thus if changing something use Tensors!
# torch.autograd.functional.jacobian jacobian from a function and tensor
class RobustKalman():
    
    def __init__(self, SysModel, test_data, c : float = 1e-3, hard_coded: bool = False, use_nn: bool = False):
        
        self.use_nn = use_nn
        if self.use_nn:
            print("Using neural network")
            self.model = SysModel  # Store the system model for f and h
            self.nn = RT_KalmanNet_nn(self.model.n, 10, [10], 1)
            """
            self.x0 = torch.transpose(SysModel.m1x_0, 0, 1)  # x0 row vector (1,2)
            self.Q = SysModel.Q
            self.R = SysModel.R
            self.T = SysModel.T  # The number of samples in the test data
            self.y = test_data
            self.c = torch.tensor(c)
            print(self.c)
            self.hard_coded = hard_coded
        
            # Preallocation of memory
            self.n = self.Q.shape[0]  # state dimension
            self.p = self.R.shape[0]  # output dimension
        
            self.Xn = torch.zeros(self.n, self.T, dtype=torch.float32)  # \hat x_t|t
            
            self.V = torch.zeros(self.n, self.n, self.T, dtype=torch.float32)  # V_t
            self.V[:, :, 0] = 1e-3 * torch.eye(self.n, dtype=torch.float32)  # Ensure float type consistency
        
            self.A = torch.zeros(self.n, self.n, self.T, dtype=torch.float32)  # Jacobian of state equation
            self.C = torch.zeros(self.p, self.n, self.T, dtype=torch.float32)  # Jacobian of output equation
            self.G = torch.zeros(self.n, self.p, self.T, dtype=torch.float32)
        
            self.th = torch.zeros(self.T, dtype=torch.float32, requires_grad=True)
        
            self.Xrekf = torch.zeros(self.n, self.T, dtype=torch.float32)  # \hat x_t
            self.Xrekf[:, 0] = self.x0
            """

        #else:
            print("Standard version")
            self.model = SysModel # Store the system model for f and h
            self.x0 = torch.transpose(SysModel.m1x_0, 0, 1) # x0 row vector (1,2)
            self.Q = SysModel.Q
            self.R = SysModel.R
            self.T = SysModel.T # The number of samples in the test data
            self.y = test_data
            self.c = torch.tensor(c)
            print(self.c)
            self.hard_coded = hard_coded
            
            # Preallocation of memory
            self.n = torch.Tensor.numpy(self.Q).shape[0] #state dimension
            self.p = torch.Tensor.numpy(self.R).shape[0] #output dimension
            
            self.Xn = torch.zeros(self.n, self.T) #allocation of memory to save \hat x_t|t
            
            self.V = torch.zeros(self.n, self.n, self.T) #allocation of memory to save V_t
            self.V[:,:,0] = 1e-3*torch.eye(2, 2)
            
            self.A = torch.zeros(self.n, self.n, self.T) #allocation of memory to save the linearizatio of state equation
            self.C = torch.zeros(self.p, self.n, self.T) #allocation of memory to save the linearizatio of output equation
            self.G = torch.zeros(self.n, self.p, self.T)
            self.th = torch.zeros(self.T) #allocation of memory to save the values of theta
            
            self.Xrekf = torch.zeros(self.n, self.T) #allocation of memory to save \hat x_t
            self.Xrekf[:, 0] = self.x0

        
    # Numerical Jacobian Computation (This is important for us since we are using the non-linear model)
    def fnComputeJacobianF(self, x_n_temp):
        # Hard coded version for translation debugging
        if self.hard_coded:
            f_jac = torch.tensor([[1/10, 1-(torch.sin(x_n_temp[1]/10)/10)],[0, 49/50]])
        else:
            f_jac = torch.autograd.functional.jacobian(self.model.f, x_n_temp)
        return f_jac
    
    def fnComputeJacobianH(self, x_rekf_temp):
        # Hard coded version for translation debugging
        if self.hard_coded:
            h_jac = torch.tensor([[1-2*x_rekf_temp[0], 2*x_rekf_temp[1]-1]])
        else:
            h_jac = torch.autograd.functional.jacobian(self.model.h,x_rekf_temp)
        return h_jac
       
    def fnComputeTheta(self, P_pred,c_t):
        # NOTE: In the computation of theta there seems to be something wrong!
        print("computing theta")
        value = torch.tensor([1])
        t1 = torch.tensor([0])
        e = torch.linalg.eig(P_pred)[0]
        r = torch.max(torch.abs(e))
        t2 = (1-1e-5)*(torch.pow(r,-1))
        
        count = 0
        while torch.abs(value) >= 1e-5:
            count = count + 1 
            theta = 0.5*(t1+t2)
            value = torch.trace(torch.linalg.solve(torch.eye(self.n) - theta * P_pred,torch.eye(self.n))-torch.eye(self.n)) + torch.log(torch.det(torch.eye(self.n) - theta * P_pred)) - c_t
            if value > 0:
                t2 = theta
            else:
                t1 = theta
            if count % 100 == 0:
                print(torch.abs(value))
                count = 0
        
        return theta
    
    # Computation of the REKF
    def fnREKF(self):
        for i in range(0, self.T - 1):
            #print(i)
            # C_t
            self.C[:, :, i] = self.fnComputeJacobianH(self.Xrekf[:,i])
            
            # L_t
            L = self.V[:,:,i] @ torch.transpose(self.C[:,:,i], 0, 1) @ torch.Tensor.inverse(self.C[:,:,i] @ self.V[:, :, i] @ torch.transpose(self.C[:,:,i], 0, 1) + self.R)
            
            # h(\hat x_t)
            hn = self.model.h(self.Xrekf[:,i])
            
            # \hat x_t|t
            self.Xn[:, i] = self.Xrekf[:, i] + torch.squeeze(L * (self.y[:,i] - hn))
            
            # A_t
            self.A[:, :, i] = self.fnComputeJacobianF(self.Xn[:,i])
            
            # G_t
            self.G[:, :, i] = self.A[:, :, i] @ L
            
            # \hat x_t+1
            self.Xrekf[:, i+1] = torch.squeeze(self.model.f(self.Xn[:, i]))
            #print("Xrekf=",self.Xrekf[:, i+1].requires_grad)
            #print(self.Xrekf[:, i+1].requires_grad)
            
            # P_t+1 - The massive fucking riccatti equation
            P = self.A[:, :, i] @ self.V[:, :, i] @ torch.transpose(self.A[:, :, i], 0, 1) - self.A[:, :, i] @ self.V[:, :, i] @ torch.transpose(self.C[:, :, i], 0, 1) @ torch.Tensor.inverse(self.C[:, :, i] @ self.V[:, :, i] @ torch.transpose(self.C[:, :, i], 0, 1) + self.R) @ self.C[:, :, i] @ self.V[:, :, i] @ torch.transpose(self.A[:, :, i], 0, 1) + self.Q
            #print("P=",P.requires_grad)
            #print((self.y[:,i] - hn).shape)            
            
            if self.use_nn:
                #print("useNN")
                c_net = self.nn(self.y[:,i] - hn) #remember that we need a row vector here with dimension (1,input_size_fcl = dimension of the output for us)
                #print("C_net=",c_net)
                #print("c_net",c_net.requires_grad)
                self.th[i] = self.fnComputeTheta(P,c_net)
                #print("th=",self.th[i].requires_grad)
                #print("yes NNN")    
            else: #here we don't use the neural network to retrieve the value of 'c'
                # th_t
                self.th[i] = self.fnComputeTheta(P,self.c)
                #print("no NNN")
            
            # V_t+1
            self.V[:, :, i+1] = torch.Tensor.inverse(torch.Tensor.inverse(P) - self.th[i] * torch.eye(self.n))
            #print("V=",self.V[:, :, i+1].requires_grad)
            
        return [self.Xrekf, self.V]
    





