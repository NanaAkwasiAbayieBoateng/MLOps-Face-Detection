#!/usr/bin/env python

import sys
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import datasets, models, transforms
import numpy as np

#DEfine the model
class Network(nn.Module):
    def __init__(self,num_classes=136):
        super().__init__()
        self.model_name='resnet18'
        self.model=models.resnet18()
        self.model.conv1=nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc=nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        x=self.model(x)
        return x

#Helper functions

def print_overwrite(step, total_step, loss, operation):
    sys.stdout.write('\r')
    if operation == 'train':
        sys.stdout.write("Train Steps: %d/%d  Loss: %.4f " % (step, total_step, loss))   
    else:
        sys.stdout.write("Valid Steps: %d/%d  Loss: %.4f " % (step, total_step, loss))
        
    sys.stdout.flush()
    

#Train (package as a function)
def landmark_train(PATH,train_loader,valid_loader):
    torch.autograd.set_detect_anomaly(True)
    network = Network()
    network.cuda()    
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(network.parameters(), lr=0.0001)
    
    loss_min = np.inf
    num_epochs = 3
    
    
    for epoch in range(1,num_epochs+1):
        
        loss_train = 0
        loss_valid = 0
        running_loss = 0
        
        network.train()
        for step in range(1,51):
        
            images, landmarks = next(iter(train_loader))
            
            images = images.cuda()
            landmarks = landmarks.view(landmarks.size(0),-1).cuda() 
            
            predictions = network(images)
            
            # clear all the gradients before calculating them
            optimizer.zero_grad()
            
            # find the loss for the current step
            loss_train_step = criterion(predictions, landmarks)
            
            # calculate the gradients
            loss_train_step.backward()
            
            # update the parameters
            optimizer.step()
            
            loss_train += loss_train_step.item()
            running_loss = loss_train/step
            
            print_overwrite(step, 50, running_loss, 'train')#len(train_loader), running_loss, 'train')
            
        network.eval() 
        with torch.no_grad():
            
            for step in range(1,51):
                
                images, landmarks = next(iter(valid_loader))
            
                images = images.cuda()
                landmarks = landmarks.view(landmarks.size(0),-1).cuda()
            
                predictions = network(images)
    
                # find the loss for the current step
                loss_valid_step = criterion(predictions, landmarks)
    
                loss_valid += loss_valid_step.item()
                running_loss = loss_valid/step
    
                print_overwrite(step, 50, running_loss, 'valid')#len(valid_loader), running_loss, 'valid')
        
        loss_train /= 50 #len(train_loader)
        loss_valid /= 50 #len(valid_loader)
        
        print('\n--------------------------------------------------')
        print('Epoch: {}  Train Loss: {:.4f}  Valid Loss: {:.4f}'.format(epoch, loss_train, loss_valid))
        print('--------------------------------------------------')
        
        if loss_valid < loss_min:
            loss_min = loss_valid
            torch.save(network.state_dict(), PATH+'face_landmarks.pth') 
            print("\nMinimum Validation Loss of {:.4f} at epoch {}/{}".format(loss_min, epoch, num_epochs))
            print('Model Saved\n')
         
    print('Training Complete')
    return network
    
    
def return_pre_trained_network():
    network=Network()
    return network
