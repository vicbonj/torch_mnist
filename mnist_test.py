from net.networks import LeNet5
import numpy as np
import keras
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from tqdm import tqdm

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train_cnn = keras.utils.to_categorical(y_train, 10)
y_test_cnn = keras.utils.to_categorical(y_test, 10)
x_train_cnn = np.expand_dims(x_train, -1)
x_test_cnn = np.expand_dims(x_test, -1)

data_train_loader = DataLoader((x_train_cnn, y_train_cnn), batch_size=256, shuffle=True, num_workers=8)
data_test_loader = DataLoader((x_test_cnn, y_test_cnn), batch_size=1024, num_workers=8)

net = LeNet5().type(torch.cuda.FloatTensor)
if torch.cuda.device_count() > 1:
    net = nn.DataParallel(net)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=2e-3)

epochs = 10

for e in range(epochs):
    print('Epoch {}'.format(e))
    for i, (images, labels) in tqdm(enumerate(data_train_loader)):
        images.type(torch.cuda.FloatTensor)
        labels.type(torch.cuda.FloatTensor)
        
        optimizer.zero_grad()
        output = net(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
