import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d
from torch.nn import MaxPool1d
from torch.nn import Flatten
from torch.nn import Linear


class Net7(nn.Module):
    def __init__(self):
        super(Net7, self).__init__()
        self.fc1 = nn.Linear(3, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 200)
        self.fc4 = nn.Linear(200, 200)
        self.fc5 = nn.Linear(200, 200)
        self.fc6 = nn.Linear(200, 200)
        self.fc7 = nn.Linear(200, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)
        return x


class Net12(nn.Module):
    def __init__(self):
        super(Net12, self).__init__()
        self.fc1 = nn.Linear(3, 200)

        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 200)
        self.fc4 = nn.Linear(200, 200)
        self.fc5 = nn.Linear(200, 200)
        self.fc6 = nn.Linear(200, 200)

        self.fc7 = nn.Linear(200, 200)
        self.fc8 = nn.Linear(200, 200)
        self.fc9 = nn.Linear(200, 200)
        self.fc10 = nn.Linear(200, 200)
        self.fc11 = nn.Linear(200, 200)

        self.fc12 = nn.Linear(200, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))

        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = F.relu(self.fc9(x))
        x = F.relu(self.fc10(x))
        x = F.relu(self.fc11(x))

        x = self.fc12(x)
        return x


class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.input_layer = Conv1d(3, 10, 1)
        self.max_pooling_layer = MaxPool1d(1)
        self.conv_layer = Conv1d(10, 50, 1)
        self.flatten_layer = Flatten()
        self.linear_layer = Linear(50, 50)
        self.output_layer = Linear(50, 1)

    def forward(self, x):
        if len(x.size()) == 1:
            n = 1

        else:
            n = x.size()[0]

        x = x.reshape((n, 3, 1))

        x = F.relu(self.input_layer(x))
        x = self.max_pooling_layer(x)
        x = F.relu(self.conv_layer(x))
        x = self.flatten_layer(x)
        x = self.linear_layer(x)
        x = self.output_layer(x)
        return x
