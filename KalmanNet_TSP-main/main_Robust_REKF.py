#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
import torch
import Simulations.config as config
import matplotlib.pyplot as plt
import math

from datetime import datetime
from Simulations.Synthetic_NL_model.parameters import Q_structure, R_structure,m,n,m1x_0,m2x_0, \
    f,h
from Simulations.Extended_sysmdl import SystemModel
from Simulations.utils import DataGen
from RobustKalmanPY.robust_kalman import RobustKalman

# KalmanNet
from Pipelines.Pipeline_EKF import Pipeline_EKF
from KNet.KalmanNet_nn import KalmanNetNN

print("Pipeline Start")
################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)

###################
###  Settings   ###
###################
args = config.general_settings()

### dataset parameters
# N is the number of sequences to be generated
args.N_E = 1 #length of training dataset 
args.N_CV = 1 #length of validation dataset 
args.N_T = 1 #length of test dataset 
args.T = 100 #input sequence length 
args.T_test = 100 #input test sequence length 

### training parameters 
args.use_cuda = False # use GPU or not (True = use GPU)
args.n_steps =  200 #number of training steps (default: 1000)
args.n_batch = 30 #input batch size for training (default: 20)
args.lr = 1e-3 #learning rate (default: 1e-3)
args.wd = 1e-3 #weight decay (default: 1e-4)

path_results = 'KNet/'
DatafolderName = 'Simulations/Synthetic_NL_model/data' + '/'

# noise q and r
r2 = torch.tensor([0.1]) # [100, 10, 1, 0.1, 0.01]
vdB = -20 # ratio v=q2/r2
v = 10**(vdB/10) #vdb = 10*log(q2/r2)
q2 = torch.mul(v,r2) #see paper pag. 8 

Q = q2[0] * Q_structure #defining the transition matrix noise
R = r2[0] * R_structure #defining the output matrix noise

print("1/r2 [dB]: ", 10 * torch.log10(1/r2[0]))
print("1/q2 [dB]: ", 10 * torch.log10(1/q2[0]))

traj_resultName = ['traj_synNL_T100.pt']
dataFileName = ['data_synNL_T100.pt'] #used to load data below
    
device = torch.device('cpu')

#########################################
###  Generate and load data DT case   ###
#########################################

sys_model = SystemModel(f, Q, h, R, args.T, args.T_test, m, n)
sys_model.InitSequence(m1x_0, m2x_0)# x0 and P0

## IMPORTANT! To compute the hard coded jacobian the sys_model also must contain the parameters of f and h
sys_model.alpha = 0.9
sys_model.beta = 1.1
sys_model.phi = 0.1*math.pi
sys_model.delta = 0.01
sys_model.a = 1
sys_model.b = 1
sys_model.c = 0

print("Start Data Gen")
DataGen(args, sys_model, DatafolderName + dataFileName[0])
print("Data Load")
print(dataFileName[0])
[train_input,train_target, _, _, _, _,_,_,_] =  torch.load(DatafolderName + dataFileName[0], map_location=device)
print("trainset input size:",train_input.size())
print("trainset target size:",train_target.size())
#%% Check of dimensions of inputs and outputs - Data Preprocessing

train_input = torch.squeeze(train_input) 
train_target = torch.squeeze(train_target)
print("trainset input size:",train_input.size())
print("trainset target size:",train_target.size())

#%% Test plot of data

plt.figure()

plt.subplot(211)
plt.plot(torch.transpose(train_target,0, 1))
plt.title("X_n")

plt.subplot(212)
plt.plot(torch.transpose(train_input,0, 1))
plt.title("Y_n")

plt.show()

# %% test of REKF on a test sample

sys_model.m1x_0 = torch.zeros(m,1)
REKF = RobustKalman(sys_model, train_input, 1e-3, True, False)

[Xrekf, _] = REKF.fnREKF()

#%% Test plot of data

plt.figure()

plt.subplot(211)
plt.plot(torch.squeeze(torch.transpose(train_target,0, 1)).numpy()[20:-1, :],color ='red' , label='train target')
plt.plot(torch.squeeze(torch.transpose(Xrekf, 0, 1)).numpy()[20:-1, :],color = 'blue' , label = 'estimated values')
plt.title("X_n")

plt.subplot(212)
plt.plot(torch.transpose(train_input,0, 1))
plt.title("Y_n")

plt.show()

#%% ##################### FROM HERE WE TEST THE IMPLEMENTATION OF RT-KalmanNet #########################
import torch.optim as optim
import torch.nn as nn

sys_model.m1x_0 = torch.zeros(m,1)
RT_KalmanNet = RobustKalman(sys_model, train_input,1e-3,True,True)
model = RT_KalmanNet

# Learning rate and number of epochs
lr = 0.01 
epochs = 50 

optimizer = optim.Adam(RT_KalmanNet.nn.parameters(), lr=lr)
criterion = nn.MSELoss()  

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")

    # Generate data
    DataGen(args, sys_model, DatafolderName + dataFileName[0])
    [train_input, train_target, _, _, _, _, _, _, _] = torch.load(DatafolderName + dataFileName[0], map_location=device)

    # Preprocess data
    train_input = torch.squeeze(train_input)  
    train_target = torch.squeeze(train_target)

    # Zeroing gradients
    optimizer.zero_grad()

    #change the input
    RT_KalmanNet.y = train_input

    # Forward pass
    [Xrekf, _] = RT_KalmanNet.fnREKF()
    Xrekf = Xrekf[:, 1:]

    # Check gradient flow
    #print("requires_grad:", Xrekf.requires_grad)

    # Compute loss
    loss = criterion(Xrekf, train_target)
    print(f"Loss: {loss.item()}")

    # Backward pass
    loss.backward(retain_graph=True )

    # Optimization step
    optimizer.step()
    #print("Optimizer step completed")

    #print("Epoch completed\n")

print("Training finished")
#%% ##################### TEST OF RT-KalmanNet #########################

DataGen(args, sys_model, DatafolderName + dataFileName[0])
[train_input, train_target, _, _, _, _, _, _, _] = torch.load(DatafolderName + dataFileName[0], map_location=device)

# Preprocess data - removing unecessary dimensions
train_input = torch.squeeze(train_input)  
train_target = torch.squeeze(train_target)

# Compute the RT-KalmaNet prediction
RT_KalmanNet.y = train_input
[Xrekf, _] = RT_KalmanNet.fnREKF()
Xrekf = Xrekf[:, 1:].detach()

# Plot for comparison
plt.figure()

plt.subplot(211)
plt.plot(torch.transpose(train_target,0, 1),color ='red' , label='train target')
plt.plot(torch.transpose(Xrekf, 0, 1),color = 'blue' , label = 'estimated values')
plt.title("X_n")

plt.show()

#%% ##################### KalmanNet #########################
# Needs to be adjusted for our purposes!
## Build Neural Network
print("KalmanNet start")
KalmanNet_model = KalmanNetNN()
KalmanNet_model.NNBuild(sys_model, args)

## Train Neural Network
KalmanNet_Pipeline = Pipeline_EKF(strTime, "KNet", "KalmanNet")
KalmanNet_Pipeline.setssModel(sys_model)
KalmanNet_Pipeline.setModel(KalmanNet_model)
print("Number of trainable parameters for KNet:",sum(p.numel() for p in KalmanNet_model.parameters() if p.requires_grad))
KalmanNet_Pipeline.setTrainingParams(args) 
print("Composition Loss:",args.CompositionLoss)
[MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = KalmanNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results)

## Test Neural Network
[MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,knet_out,RunTime] = KalmanNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results)
