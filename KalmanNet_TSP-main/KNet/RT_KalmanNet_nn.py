# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 09:42:01 2025
In this file we build the neural network presented at pag. 5 in the presentation's slide
It is composed by 
1) a fully connected layer with two layer
2) a DNN with feedback
"""

import torch
import torch.nn as nn


class RT_KalmanNet_nn(nn.Module):
    def __init__(self, input_size_fcl,output_size_fcl, hidden_sizes, output_size_DNN):
        """
        Args:
            input_size (int): Numero di neuroni in ingresso al Fully Connected Layer.
            hidden_sizes (list of int): Lista con il numero di neuroni per ogni hidden layer della DNN.
            output_size (int): Numero di neuroni in uscita dalla DNN.
        """
        super().__init__()

        #Creation of the fully connected layer
        self.fcl = nn.Linear(input_size_fcl, output_size_fcl)

        #First layer of the DNN (it takes output of the fc layer + previous output of the DNN)
        self.dnn_input_layer = nn.Linear(output_size_fcl + output_size_DNN, hidden_sizes[0])

        #Other layer of the DNN
        self.dnn_hidden_layers = nn.ModuleList()#preparing the list to save other layer
        for i in range(len(hidden_sizes) - 1): #adding the various layers
            self.dnn_hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))

        #output layer of the DNN
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size_DNN)

        #initial state of the DNN
        self.previous_output = torch.zeros(1,output_size_DNN)  # Assume batch_size=1, change if it necessary

    def forward(self, x):
        """
        Args:
            x (Tensor): input of the network (expected row vector of dimension input_size_fcl)

        Returns:
            Tensor: Output of DNN.
        """
        # Fully Connected Layer
        x = torch.relu(self.fcl(x))

        #print("X_shape",x.shape)
        #print("pre_out_shape",self.previous_output.shape)
        #input("out fully")
        # Combining the previous output and the output of the fully connected layer
        x_combined = torch.cat((x, self.previous_output), dim=1)
        #print("x_combined_shape",x_combined.shape)
        #input("entering dnn")
        dnn_output = torch.relu(self.dnn_input_layer(x_combined))

        # Passing through the hidden layers
        for layer in self.dnn_hidden_layers:
            dnn_output = torch.relu(layer(dnn_output))
        #input("entering output layer dnn")
        # output layer
        final_output = torch.sigmoid(self.output_layer(dnn_output)) #I apply exp to guarantee that the value c will be positive
        #input("exiting output layer dnn")
        # Update the previous output
        self.previous_output = final_output.detach() # disable backpropagation for this variable

        return final_output
