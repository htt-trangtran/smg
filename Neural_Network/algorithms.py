############################
# written by Trang H. Tran and Lam M. Nguyen
############################

"""
Algorithms
"""

import torch 
import math
import pandas as pd
import random
from record_history import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def SMG_train (NameNet, num_epoch, loader, test_loader, scheduleLR, beta, criterion, record_path, record_name):

    #---------------------------------------------------------------------------
    print('Step 1: Restart network with starting learning rate', scheduleLR(1))
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    net = NameNet() 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device) 
    
    # Initialize to save data
    record_history ('initial', 0, net, loader, test_loader, criterion, record_path, record_name)
    seed_alg = random.randint(0,100)
    torch.manual_seed(seed_alg)

    # Initialize 
    list_m = {k: torch.zeros_like(f.data, memory_format=torch.preserve_format) for k, f in net.named_parameters()}
    list_v = {k: torch.zeros_like(f.data, memory_format=torch.preserve_format) for k, f in net.named_parameters()}

    #---------------------------------------------------------------------------
    print('Step 2: Train the network', num_epoch[0], 'epoch(s) for', record_name)
    
    for epoch in range (num_epoch[0]):
        lr = scheduleLR(epoch + 1)
        # Reset list v
        list_m = {k: list_v[k] for k, f in net.named_parameters()}
        list_v = {k: torch.zeros_like(f.data, memory_format=torch.preserve_format) for k, f in net.named_parameters()}

        # Start training for an epoch ------------------------------------------        
        for i, data in enumerate (loader, 0):
            # Get the inputs and labels ----------------------------------------
            inputs, labels = data[0].to(device), data[1].to(device)
            
            # Zero grad all the trainable parameters ---------------------------
            net.zero_grad() 

            # Forward pass and backward pass -----------------------------------
            outputs = net(inputs)
            loss = criterion (outputs, labels)
            loss.backward() 

            # Loop through the weights -----------------------------------------
            for name, f in net.named_parameters():
                # Update rule
                with torch.no_grad():
                    normalize = labels.size(0) / len(loader.dataset)
                    list_v[name].add_(f.grad.data, alpha = normalize)
                    update =  torch.add( torch.mul(list_m[name], beta), f.grad.data, alpha = 1 - beta)
                    f.data.add_(update, alpha=-lr)

        # Measure and add statistics after a number of epoch --------------------
        if (epoch+1) % (num_epoch[1]) == 0:
            record_history ('measure', epoch + 1, net, loader, test_loader, criterion, record_path, record_name)
    #---------------------------------------------------------------------------
    print('Step 3: Save the results') 
    record_history ('save', 0, net, loader, test_loader, criterion, record_path, record_name)
    print('Finish Training \n ------------------------------------------------')

#-------------------------------------------------------------------------------
def SGD_train (NameNet, num_epoch, loader, test_loader, scheduleLR, criterion, record_path, record_name):

    #---------------------------------------------------------------------------
    print('Step 1: Restart network with starting learning rate', scheduleLR(1))
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    net = NameNet() 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device) 
    
    # Initialize to save data
    record_history ('initial', 0, net, loader, test_loader, criterion, record_path, record_name)
    seed_alg = random.randint(0,100)
    torch.manual_seed(seed_alg)

    #---------------------------------------------------------------------------
    print('Step 2: Train the network', num_epoch[0], 'epoch(s) for', record_name)
    
    for epoch in range (num_epoch[0]):
        lr = scheduleLR(epoch + 1)

        # Start training for an epoch ------------------------------------------        
        for i, data in enumerate (loader, 0):
            # Get the inputs and labels ----------------------------------------
            inputs, labels = data[0].to(device), data[1].to(device)
            
            # Zero grad all the trainable parameters ---------------------------
            net.zero_grad() 

            # Forward pass and backward pass -----------------------------------
            outputs = net(inputs)
            loss = criterion (outputs, labels)
            loss.backward() 
            
            # Loop through the weights -----------------------------------------
            for name, f in net.named_parameters():
                # Update rule
                f.data.add_(f.grad.data, alpha = -lr)

        # Measure and add statistics after a number of epoch --------------------
        if (epoch+1) % (num_epoch[1]) == 0:
            record_history ('measure', epoch + 1, net, loader, test_loader, criterion, record_path, record_name)
    #---------------------------------------------------------------------------
    print('Step 3: Save the results') 
    record_history ('save', 0, net, loader, test_loader, criterion, record_path, record_name)
    print('Finish Training \n ------------------------------------------------')

#-------------------------------------------------------------------------------

def SGDM_train (NameNet, num_epoch, loader, test_loader, scheduleLR, beta, criterion, record_path, record_name):

    #---------------------------------------------------------------------------
    print('Step 1: Restart network with starting learning rate', scheduleLR(1))
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    net = NameNet() 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device) 
    
    # Initialize to save data
    record_history ('initial', 0, net, loader, test_loader, criterion, record_path, record_name)
    seed_alg = random.randint(0,100)
    torch.manual_seed(seed_alg)

    # Initialize for SGD M
    list_m = {k: torch.zeros_like(f.data, memory_format=torch.preserve_format) for k, f in net.named_parameters()}

    #---------------------------------------------------------------------------
    print('Step 2: Train the network', num_epoch[0], 'epoch(s) for', record_name)
    
    for epoch in range (num_epoch[0]):
        lr = scheduleLR(epoch + 1)

        # Start training for an epoch ------------------------------------------        
        for i, data in enumerate (loader, 0):
            # Get the inputs and labels ----------------------------------------
            inputs, labels = data[0].to(device), data[1].to(device)
            
            # Zero grad all the trainable parameters ---------------------------
            net.zero_grad() 

            # Forward pass and backward pass -----------------------------------
            outputs = net(inputs)
            loss = criterion (outputs, labels)
            loss.backward() 

            # Loop through the weights -----------------------------------------
            for name, f in net.named_parameters():
                # Update rule
                with torch.no_grad():
                    list_m[name].mul_(beta).add_(f.grad.data)
                    f.data.add_(list_m[name], alpha=-lr)

        # Measure and add statistics after a number of epoch --------------------
        if (epoch+1) % (num_epoch[1]) == 0:
            record_history ('measure', epoch + 1, net, loader, test_loader, criterion, record_path, record_name)
    #---------------------------------------------------------------------------
    print('Step 3: Save the results') 
    record_history ('save', 0, net, loader, test_loader, criterion, record_path, record_name)
    print('Finish Training \n ------------------------------------------------')

#-------------------------------------------------------------------------------
def ADAM_train (NameNet, num_epoch, loader, test_loader, scheduleLR, criterion, record_path, record_name):

    #---------------------------------------------------------------------------
    print('Step 1: Restart network with starting learning rate', scheduleLR(1))
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    net = NameNet() 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device) 
    
    # Initialize to save data
    record_history ('initial', 0, net, loader, test_loader, criterion, record_path, record_name)
    seed_alg = random.randint(0,100)
    torch.manual_seed(seed_alg)

    # Initialize for Adam
    num_params = len(list(net.parameters() ))
    eps = 1e-8
    t = 0
    beta_1 = 0.9
    beta_2 = 0.999
    list_m = {k: torch.zeros_like(f.data, memory_format=torch.preserve_format) for k, f in net.named_parameters()}
    list_v = {k: torch.zeros_like(f.data, memory_format=torch.preserve_format) for k, f in net.named_parameters()}

    #---------------------------------------------------------------------------
    print('Step 2: Train the network', num_epoch[0], 'epoch(s) for', record_name)
    
    for epoch in range (num_epoch[0]):
        lr = scheduleLR(epoch + 1)

        # Start training for an epoch ------------------------------------------        
        for i, data in enumerate (loader, 0):
            # Get the inputs and labels ----------------------------------------
            inputs, labels = data[0].to(device), data[1].to(device)
            
            # Zero grad all the trainable parameters ---------------------------
            net.zero_grad() 

            # Forward pass and backward pass -----------------------------------
            outputs = net(inputs)
            loss = criterion (outputs, labels)
            loss.backward() 
            
            # Update the number of steps ---------------------------------------
            t += 1
            bias_correction1 = 1 - beta_1 ** t
            bias_correction2 = 1 - beta_2 ** t

            # Loop through the weights -----------------------------------------
            for name, f in net.named_parameters():
                # Update rule
                with torch.no_grad():
                    list_m[name].mul_(beta_1).add_(f.grad.data, alpha = 1 - beta_1)
                    list_v[name].mul_(beta_2).addcmul_(f.grad.data, f.grad.data, value = 1 - beta_2)
                    denom = (list_v[name].sqrt() / math.sqrt(bias_correction2)).add_(eps)
                    stepsize = lr /(bias_correction1)
                    f.data.addcdiv_(list_m[name], denom, value= - stepsize)

        # Measure and add statistics after a number of epoch --------------------
        if (epoch+1) % (num_epoch[1]) == 0:
            record_history ('measure', epoch + 1, net, loader, test_loader, criterion, record_path, record_name)
    #---------------------------------------------------------------------------
    print('Step 3: Save the results') 
    record_history ('save', 0, net, loader, test_loader, criterion, record_path, record_name)
    print('Finish Training \n ------------------------------------------------')

