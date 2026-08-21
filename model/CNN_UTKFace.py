import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_UTKFace(nn.Module):
    def __init__(self):
        super(CNN_UTKFace, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1)  # 3 input channels, 16 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1)  # 16 input channels, 32 output channels, 3x3 kernel
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1)  # 32 input channels, 64 output channels, 3x3 kernel

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 26 * 26, 128)  # Assuming input size of 200x200 (see note below)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # Final output layer (2 classes)

        self.activation = F.relu
        self.dropout = nn.Dropout(0.0)

    def forward(self, x):
        # Convolutional layers with max pooling
        h = self.conv1(x)
        h = self.activation(h)
        h = F.max_pool2d(h, 2)

        h = self.conv2(h)
        h = self.activation(h)
        h = F.max_pool2d(h, 2)

        h = self.conv3(h)
        h = self.activation(h)
        h = F.max_pool2d(h, 2)
        h = self.dropout(h)

        # Flatten for fully connected layers
        h = torch.flatten(h, 1)

        # Fully connected layers
        h = self.fc1(h)
        h = self.activation(h)
        h = self.dropout(h)

        h = self.fc2(h)
        h = self.activation(h)
        h = self.dropout(h)

        h = self.fc3(h)
        output = F.log_softmax(h, dim=1)
        return output

    def get_last_hidden_layer(self, x):
        # Convolutional layers with max pooling
        h = self.conv1(x)
        h = self.activation(h)
        h = F.max_pool2d(h, 2)

        h = self.conv2(h)
        h = self.activation(h)
        h = F.max_pool2d(h, 2)

        h = self.conv3(h)
        h = self.activation(h)
        h = F.max_pool2d(h, 2)

        # Flatten for fully connected layers
        h = torch.flatten(h, 1)

        # Fully connected layers up to the last hidden layer
        h = self.fc1(h)
        h = self.activation(h)
        output = self.fc2(h)
        return output.detach().cpu().numpy()


class CNN_UTKFace_dropout(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(CNN_UTKFace_dropout, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1)
        self.conv1_dropout = nn.Dropout2d(dropout_rate)

        self.conv2 = nn.Conv2d(16, 32, 3, stride=1)
        self.conv2_dropout = nn.Dropout2d(dropout_rate)

        self.conv3 = nn.Conv2d(32, 64, 3, stride=1)
        self.conv3_dropout = nn.Dropout2d(dropout_rate)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 26 * 26, 128)
        self.fc1_dropout = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(128, 64)
        self.fc2_dropout = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(64, 2)
        self.fc3_dropout = nn.Dropout(dropout_rate)

        self.activation = F.relu

    def forward(self, x):
        # Convolutional layers with max pooling and dropout
        h = self.conv1(x)
        h = self.activation(h)
        h = self.conv1_dropout(h)
        h = F.max_pool2d(h, 2)

        h = self.conv2(h)
        h = self.activation(h)
        h = self.conv2_dropout(h)
        h = F.max_pool2d(h, 2)

        h = self.conv3(h)
        h = self.activation(h)
        h = self.conv3_dropout(h)
        h = F.max_pool2d(h, 2)

        # Flatten for fully connected layers
        h = torch.flatten(h, 1)

        # Fully connected layers with dropout
        h = self.fc1(h)
        h = self.activation(h)
        h = self.fc1_dropout(h)
        h = self.fc2(h)
        h = self.activation(h)
        h = self.fc2_dropout(h)

        h = self.fc3(h)
        h = self.fc3_dropout(h)
        output = F.log_softmax(h, dim=1)
        return output

    def get_last_hidden_layer(self, x):
        # Convolutional layers with max pooling and dropout
        h = self.conv1(x)
        h = self.activation(h)
        h = self.conv1_dropout(h)
        h = F.max_pool2d(h, 2)

        h = self.conv2(h)
        h = self.activation(h)
        h = self.conv2_dropout(h)
        h = F.max_pool2d(h, 2)

        h = self.conv3(h)
        h = self.activation(h)
        h = self.conv3_dropout(h)
        h = F.max_pool2d(h, 2)

        # Flatten for fully connected layers
        h = torch.flatten(h, 1)

        # Fully connected layers up to the last hidden layer with dropout
        h = self.fc1(h)
        h = self.activation(h)
        h = self.fc1_dropout(h)

        output = self.fc2(h)
        return output.detach().cpu().numpy()


class CNN_UTKFace_dp(nn.Module):
    def __init__(self):
        super(CNN_UTKFace, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1)  # 3 input channels, 16 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1)  # 16 input channels, 32 output channels, 3x3 kernel
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1)  # 32 input channels, 64 output channels, 3x3 kernel

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 26 * 26, 128)  # Assuming input size of 200x200 (see note below)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # Final output layer (2 classes)

        self.activation = F.relu
        self.dropout = nn.Dropout(0.0)

    def forward(self, x):
        # Convolutional layers with max pooling
        h = self.conv1(x)
        h = self.activation(h)
        h = F.max_pool2d(h, 2)

        h = self.conv2(h)
        h = self.activation(h)
        h = F.max_pool2d(h, 2)

        h = self.conv3(h)
        h = self.activation(h)
        h = F.max_pool2d(h, 2)
        h = self.dropout(h)

        # Flatten for fully connected layers
        h = torch.flatten(h, 1)

        # Fully connected layers
        h = self.fc1(h)
        h = self.activation(h)
        h = self.dropout(h)

        h = self.fc2(h)
        h = self.activation(h)
        h = self.dropout(h)

        h = self.fc3(h)
        output = F.log_softmax(h, dim=1)
        return output

    def get_last_hidden_layer(self, x):
        # Convolutional layers with max pooling
        h = self.conv1(x)
        h = self.activation(h)
        h = F.max_pool2d(h, 2)

        h = self.conv2(h)
        h = self.activation(h)
        h = F.max_pool2d(h, 2)

        h = self.conv3(h)
        h = self.activation(h)
        h = F.max_pool2d(h, 2)

        # Flatten for fully connected layers
        h = torch.flatten(h, 1)

        # Fully connected layers up to the last hidden layer
        h = self.fc1(h)
        h = self.activation(h)
        output = self.fc2(h)
        return output.detach().cpu().numpy()
