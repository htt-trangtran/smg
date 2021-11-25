############################
# written by Trang H. Tran and Lam M. Nguyen
############################

"""
record_history 

There are three phases to record the training process history. 
The result will be recorded in a dictionary named statistics.
phase 'initial': set up the dictionary statistics at the beginning
phase 'measure': called each time we want to record the data to the dictionary
phase 'save'   : save the dictionary and the network when complete training
"""

import torch
import pandas as pd
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def record_history (phase, number, net, loader, test_loader, criterion, record_path, record_name):
    # Initialize phase:
    if (phase == 'initial'):
        global statistics
        statistics = {
        "number": [],
        "loss": [],
        "grad_loss": [],
        "acc_train": [],
        "acc_test": [] 
         }
        global total_train, total_test
        total_train, total_test = len(loader.dataset), len(test_loader.dataset)

    # Measure phase: 
    if (phase == 'measure'):
        add_loss = 0.0
        correct_train = 0.0
        correct_test = 0.0
        add_grad_loss = 0.0   # Norm of grad loss 

        # Loop through the training data ---------------------------------------
        for i, data in enumerate (loader, 0):
            # Get the inputs and labels ------------------------------------
            inputs, labels = data[0].to(device), data[1].to(device)
            
            # Zero grad all the trainable parameters -----------------------
            net.zero_grad() 

            # Forward pass and backward pass -------------------------------
            outputs = net(inputs)
            loss = criterion (outputs, labels)
            loss.backward() 
            
            with torch.no_grad():
                normalize = labels.size(0) / total_train
                # Add the loss for this data -----------------------------------
                add_loss += loss.item() * normalize

                # Add the grad loss data ---------------------------------------
                for name, f in net.named_parameters():
                    add_grad_loss += torch.sum(f.grad * f.grad).item() * normalize

                # Add the accuracy for this data -------------------------------
                _, predicted = torch.max(outputs.data, 1)
                correct_train += (predicted == labels).sum().item()

        # Loop through the test data -------------------------------------------
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = net(inputs)
                # Add the accuracy for this data -------------------------------
                _, predicted = torch.max(outputs.data, 1)
                correct_test += (predicted == labels).sum().item()        

        correct_train = 100 * correct_train / total_train
        correct_test  = 100 * correct_test / total_test

        # Add statistics --------------------------------------
        print ('--- # %d Loss %.5f Grad Loss %.5f Train & Test Acc: %.2f  %.2f' % (number, add_loss,
        add_grad_loss, correct_train, correct_test))
        
        statistics ["number"].append(number)
        statistics ["loss"].append(add_loss)
        statistics ["grad_loss"].append(add_grad_loss)
        statistics ["acc_train"].append(correct_train)
        statistics ["acc_test"].append(correct_test)
    
    # Save phase:
    if (phase == 'save'):
        # Save the current net
        torch.save(net.state_dict(), record_path + record_name)

        # Save the training process history
        record_name = record_name + '.csv'
        for key, value in statistics.items():
            pd.DataFrame(value).to_csv(record_path + str(key) + '_' + record_name, index = False)