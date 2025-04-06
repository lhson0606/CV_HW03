from torch import nn


class Net3(nn.Module):
    """
    A simple 3-layers feedforward neural network (1 input + 1 hidden + 1 output).
    """
    def __init__(self):
        super(Net3, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, image):
        # (batch_size, 28, 28) -> (batch_size, 28 * 28)
        x = self.flatten(image) # (batch_size, 28 * 28)

        # (batch_size, 28 * 28) -> (batch_size, 128)
        x = self.fc1(x)
        x = self.relu(x) # (batch_size, 128)

        # (batch_size, 128) -> (batch_size, 10)
        x = self.fc2(x) # (batch_size, 10)
        return x


class Net4(nn.Module):
    """
    A simple 4-layers feedforward neural network (1 input + 2 hidden + 1 output).
    """
    def __init__(self):
        super(Net4, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, image):
        # (batch_size, 28, 28) -> (batch_size, 28 * 28)
        x = self.flatten(image) # (batch_size, 28 * 28)

        # (batch_size, 28 * 28) -> (batch_size, 128)
        x = self.fc1(x)
        x = self.relu(x) # (batch_size, 128)

        # (batch_size, 128) -> (batch_size, 64)
        x = self.fc2(x)
        x = self.relu(x) # (batch_size, 64)

        # (batch_size, 64) -> (batch_size, 10)
        x = self.fc3(x) # (batch_size, 10)
        return x