import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
        def __init__(self, drop=0.1, activation_function= F.relu):
            """
        Neural network model with convolutional and fully connected layers.

        Args:
            drop (float, optional): Dropout probability. Default is 0.1.
            activation_function (function, optional): Activation function to use.
                Default is torch.nn.functional.relu.

        """
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5) 
            self.bn1 = nn.BatchNorm2d(6)  # Batch normalization after conv1
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.bn2 = nn.BatchNorm2d(16)  # Batch normalization after conv2
            self.fc1 = nn.Linear(16 * 9 * 9, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)
            self.dropout = nn.Dropout(p=drop)
            self.activ = activation_function
        def forward(self, x):
            """
        Perform a forward pass through the neural network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output logits tensor of shape (batch_size, num_classes).

        """
            x = self.pool(self.activ(self.bn1(self.conv1(x)))) 
            # 44*44*6 , 22*22*6 
            
            x = self.pool(self.activ(self.bn2(self.conv2(x))))
            # 18*18*16 , 9*9*16 
            
            x = torch.flatten(x, 1) # flatten all dimensions except batch
            x = self.activ(self.dropout(self.fc1(x)))
            # x = self.dropout(x)
            x = self.activ(self.dropout(self.fc2(x)))
            # x = self.dropout(x)
            x = self.fc3(x)
            # x = F.softmax(x, dim=1)
            return x