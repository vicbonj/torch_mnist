from net.networks import LeNet5
import numpy as np
import keras
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from tqdm import tqdm

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape((-1, 1, 28, 28)).astype('float32')
x_test = x_test.reshape((-1, 1, 28, 28)).astype('float32')
x_train /= 255.
x_test /= 255.
y_train = y_train.astype(np.int64)
y_test = y_test.astype(np.int64)

class TMPDataset(Dataset):

    def __init__(self, a, b):
        self.x = a
        self.y = b

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return len(self.y)

data_train_loader = DataLoader(TMPDataset(x_train, y_train), batch_size=256, shuffle=True, num_workers=8)
data_test_loader = DataLoader(TMPDataset(x_test, y_test), batch_size=1024, num_workers=8)

net = LeNet5().type(torch.cuda.FloatTensor)
if torch.cuda.device_count() > 1:
    net = nn.DataParallel(net)
criterion = nn.CrossEntropyLoss().type(torch.cuda.FloatTensor)
optimizer = optim.Adam(net.parameters(), lr=2e-3)

epochs = 10

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
for e in range(epochs):
    print('Epoch {}'.format(e))
    for i, (images, labels) in tqdm(enumerate(data_train_loader)):
        images = images.type(torch.cuda.FloatTensor)
        labels = labels.type(torch.cuda.FloatTensor)
        
        optimizer.zero_grad()
        output = net(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
