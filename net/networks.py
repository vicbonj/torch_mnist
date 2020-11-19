import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)#, padding=1, padding_mode='reflection')
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=1, padding_mode='reflection')
        self.conv3 = nn.Conv2d(16, 120, 5, padding=1, padding_mode='reflection')
        self.dense1 = nn.Linear(120, 84)
        self.dense2 = nn.Linear(84, 10)
        self.last_act = nn.LogSoftmax(dim=-1)
        
    def forward(self, x):
        print(x.size(0))
        output = self.maxpool(self.relu(self.conv1(x)))
        y = self.maxpool(self.relu(self.conv2(output)))
        output = self.maxpool(self.relu(self.conv2(output)))
        output += y
        output = self.relu(self.conv3(output))
        output = output.view(x.size(0), -1)
        output = self.relu(self.dense1(output))
        output = self.last_act(self.dense2(output))
        return output
