#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 14:17:10 2018

AI for Self Driving Car

@author: jamesponwith
"""

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# Creating the architecture of the Neural Network
# Each network object will have:
    # 5 Input Neurons
    # 30 Hidden Layer Neurons
    # 3 Output Neurons

class Network(nn.Module):
    
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size # Neural net will have 5 neurons on input layer
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_action)

    # Forward Propogation
    def forward(self, state):
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values
    
# Implementing Experience Replay
# Creating a memory length of 100 000 memories to maintain a long term memory

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
        
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory,batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)
    
# Implementing Deep Q Learning    

class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000) # 100 000 memories
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0) # Vector of 5 dimensions (the three sensors and orientation and minus orientation)
        self.last_action = 0                                    # .unsqueeze creates fake dimension to address batch 
        self.last_reward = 0
    
    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile = True))*100) # T=100
        action = probs.multinomial()
        return action.data[0,0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target) #Temporal difference (Hover loss)
        self.optimizer.zero_grad() # Now Back propogate the loss error through the network to update the weights with stocastic gradient descent
        td_loss.backward(retain_variables = True)
        self.optimizer.step() # Updates the weights for how much they had an effect on the error
        
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_reward, batch_action = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action    
        
    def score(self):
        return sum((self.reward_window)/len(self.reward_window)+1) # Plus 1 to avoid crashing system
    
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    }, 'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")