import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, 5, padding=2)
        self.maxpool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 16, 5, padding=2)
        self.dense1 = nn.Linear(784, 1024)
        self.dense2 = nn.Linear(1024, 256)
        self.dense3 = nn.Linear(256, 10)
        self.last_act = nn.Softmax(dim=1)
        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.2)
        
    def forward(self, x):
        output = self.maxpool(self.relu(self.conv1(x)))
        output = self.maxpool(self.relu(self.conv2(output)))
        output = torch.flatten(output, 1)
        output = self.relu(self.dense1(output))
        output = self.drop1(output)
        output = self.relu(self.dense2(output))
        output = self.drop2(output)
        output = self.last_act(self.dense3(output))
        return output
