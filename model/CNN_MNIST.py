import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_MNIST(nn.Module):
    def __init__(self, num_classes, latent_dim=64):
        super(CNN_MNIST, self).__init__()
        self.out_features = num_classes
        self.latent_dim = latent_dim
        # --- Convolutional part ---
                        # Note: For Qdrop quantization, the order conv->bn->relu may cause issues
        # because Qdrop expects conv->relu order and handles ReLU separately.
        # Consider using Sequential for better compatibility with quantization frameworks.
        self.conv1 = nn.Conv2d(1, 128, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(512)
        self.mp = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        # --- Fully-connected part ---
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(8192, 512)  # 32x32 after 3 maxpools: 512*4*4 = 8192
        self.fc2 = nn.Linear(512, 64)
        # Add latent representation layer for VIB
        self.fc_encode = nn.Linear(64, latent_dim * 2)
        # Output layer for standard forward
        self.fc3_standard = nn.Linear(64, num_classes)
        # Output layer for VIB forward
        self.fc3 = nn.Linear(latent_dim, num_classes)

    def encode(self, x):
        """Encode input to latent representation with mu and std"""
        # Forward through CNN layers
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.mp(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.mp(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.mp(x)
        x = x.view(-1, 8192)  # 32x32 after 3 maxpools: 512*4*4 = 8192
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        h = self.fc_encode(x)  # [batch, latent_dim * 2]
        mu, std = h[:, :self.latent_dim], h[:, self.latent_dim:]
        std = F.softplus(std - 5, beta=1)  # Ensure std is positive
        return mu, std
    def forward(self, x, use_vib=False):
        if use_vib:
            mu, std = self.encode(x)
            # Sample from latent distribution
            eps = torch.randn_like(std)
            z = mu + std * eps
            output = self.fc3(z)
        else:
            # Standard forward without VIB
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.mp(x)
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.mp(x)
            x = self.relu(self.bn3(self.conv3(x)))
            x = self.mp(x)
            x = x.view(-1, 8192)  # 32x32 after 3 maxpools: 512*4*4 = 8192
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            output = self.fc3_standard(x)
        return output

    def get_mu_std(self, x):
        """Get mu and std from latent representation"""
        return self.encode(x)



# class CNN_MNIST(nn.Module):
#     def __init__(self, num_classes, latent_dim=64):
#         super(CNN_MNIST, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.pool = nn.MaxPool2d(2, 2)
#
#         self.fc1 = nn.Linear(2048, 512)
#         self.fc2 = nn.Linear(512, latent_dim)
#         self.fc3 = nn.Linear(latent_dim, num_classes)
#
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.0)
#
#     def forward(self, x):
#         x = self.pool(self.relu(self.bn1(self.conv1(x))))
#         x = self.pool(self.relu(self.bn2(self.conv2(x))))
#         x = self.pool(self.relu(self.bn3(self.conv3(x))))
#         x = x.view(x.size(0), -1)
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

