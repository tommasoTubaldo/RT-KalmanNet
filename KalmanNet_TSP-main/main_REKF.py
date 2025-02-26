#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
import torch
import Simulations.config as config
import matplotlib.pyplot as plt


from datetime import datetime
from Simulations.REKF_model.parameters import n, m, Q, R, f, h, m1x_0, m2x_0 #(Original model parameters)
#from Simulations.Synthetic_NL_model.parameters import n, m, Q_structure, R_structure, f, h, m1x_0, m2x_0 #synthetic_NL_model
from Simulations.Extended_sysmdl import SystemModel
from Simulations.utils import DataGen
from RobustKalmanPY.robust_kalman import RobustKalman

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
# N is the number of sequences to be generated!
args.N_E = 1 #length of training dataset (I kept this length there is no specificagtion in the paper)
args.N_CV = 1 #length of validation dataset (I kept this length there is no specificagtion in the paper)
args.N_T = 1 #length of test dataset (I kept this length there is no specificagtion in the paper)
args.T = 100 #input sequence length (see paper pag.10 section c)
args.T_test = 100 #input test sequence length (I kept this length there is no specificagtion in the paper)

### training parameters (I kept all these parameters I found them in the Lorenz attractor main file)
args.use_cuda = False # use GPU or not (True = use GPU)
args.n_steps =  200 #number of training steps (default: 1000)
args.n_batch = 30 #input batch size for training (default: 20)
args.lr = 1e-3 #learning rate (default: 1e-3)
args.wd = 1e-3 #weight decay (default: 1e-4)

#offset = 0 # offset for the data
#chop = False # whether to chop data sequences into shorter sequences
path_results = 'KNet/'
DatafolderName = 'Simulations/REKF_model/data' + '/'
#flag that will be used later on in the code
#switch = 'partial' # 'full' or 'partial' or 'estH'v 

traj_resultName = ['traj_REKF_T100.pt']
dataFileName = ['data_REKF_T100.pt'] #used to load data below
    
device = torch.device('cpu')

#########################################
###  Generate and load data DT case   ###
#########################################

sys_model = SystemModel(f, Q, h, R, args.T, args.T_test, m, n)# parameters for GT (original model)
#sys_model = SystemModel(f, Q_structure, h, R_structure, args.T, args.T_test, m, n)# parameters for GT (synthetic NL model)
sys_model.InitSequence(m1x_0, m2x_0)# x0 and P0

print("Start Data Gen")
DataGen(args, sys_model, DatafolderName + dataFileName[0])
print("Data Load")
print(dataFileName[0])
[train_input,train_target, _, _, _, _,_,_,_] =  torch.load(DatafolderName + dataFileName[0], map_location=device)
print("trainset input size:",train_input.size())
print("trainset target size:",train_target.size())

train_input = torch.squeeze(train_input,1) #(for original model)
#train_input = torch.squeeze(train_input) #(for synthetic NL model)
train_target = torch.squeeze(train_target)
print("trainset input size:",train_input.size())
print("trainset target size:",train_target.size())

#print(train_input)
#print(train_target)
#print(sys_model.m1x_0)
#%% Test plot of data

plt.figure()

plt.subplot(211)
plt.plot(torch.transpose(train_target,0, 1))#exchange 0 and 1 dimension
plt.title("X_n")

plt.subplot(212)
plt.plot(torch.transpose(train_input,0, 1))
plt.title("Y_n")

# %% test of REKF on a test sample
sys_model.m1x_0 = torch.zeros(m,1)
# The test shows something is broken .. must resech tomorrow
REKF = RobustKalman(sys_model, train_input, 1e-3, True,False)
print(REKF.use_nn)

[Xrekf, _] = REKF.fnREKF()
#print(Xrekf.shape)
#print(Xrekf)


#%% Test plot of data

plt.figure()

plt.subplot(211)
plt.plot(torch.transpose(train_target,0, 1))
plt.plot(torch.transpose(Xrekf, 0, 1))
plt.title("X_n")

plt.subplot(212)
plt.plot(torch.transpose(train_input,0, 1))
plt.title("Y_n")

#%% ##################### FROM HERE WE TEST THE IMPLEMENTATION OF RT-KalmanNet #########################
RT_KalmanNet = RobustKalman(sys_model, train_input, 1e-3,True,True)
#Note: as a first implementation I will make training and test here in the main file 
############IMPORTING USEFUL LIBRARIES###############
import torch.optim as optim
import torch.nn as nn

##############TRAINING#####################
lr = 0.01;#defining the learning rate
epochs = 20;
input()
optimizer = optim.Adam(RT_KalmanNet.nn.parameters(), lr=lr)
criterion = nn.MSELoss()  #loss criterion, minimize the error on the state
for epoch in range(epochs):
    
    #generate and load data
    DataGen(args, sys_model, DatafolderName + dataFileName[0])
    [train_input,train_target, _, _, _, _,_,_,_] =  torch.load(DatafolderName + dataFileName[0], map_location=device)
    train_input = torch.squeeze(train_input,1) #reduce dimension
    train_target = torch.squeeze(train_target)
    
    #reset the state
    RT_KalmanNet.x0 = torch.zeros(m,1)
    
    #change the input of the object with the new generated data
    RT_KalmanNet.y = train_input
    
    #Forward pass
    [Xrekf, _] = RT_KalmanNet.fnREKF()
    Xrekf.requires_grad = True
    input("fine")
   
    # Compute loss
    loss = criterion(Xrekf, train_target)
    
    print(Xrekf.requires_grad)  # Deve restituire True, altrimenti il backprop non funziona
    
    input()
    
    # Backpropagation
    optimizer.zero_grad()  # Reset gradients
    loss.backward()  # Compute gradients
    optimizer.step()  # Update network weights
    
    #print(Xrekf.shape)
    #print(train_target.shape)
    #input()
    print(f"Epoch {epoch}, Loss: {loss.item()}")
    input()
    
    
    
print("done");


