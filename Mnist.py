import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import numpy as np

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, 
                            transform=transforms.ToTensor(),download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', 
                           train=False, transform=transforms.ToTensor())

batch_size = 100
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

'''
class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_dim, output_dim)  
    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        return out
'''
    
input_dim = 28*28
hidden_dim = 100
output_dim = 10
net = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
