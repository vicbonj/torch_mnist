import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 120, 5)
        self.dense1 = nn.Liner(120, 84)
        self.dense2 = nn.Linear(84, 10)
        self.last_act = nn.LogSoftmax(dim=-1)
        
    def forward(self, x):
        output = self.maxpool(self.relu(self.conv1(x)))
        x = self.maxpool(self.relu(self.conv2(output)))
        output = self.maxpool(self.relu(self.conv2(output)))
        output += x
        output = self.relu(self.conv3(output))
        output = output.view(img.size(0), -1)
        output = self.relu(self.dense1(output))
        output = self.last_act(self.dense2(output))
        return output
