import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import pandas as pd
import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F
import torch.mps 
import itertools
import csv
from torchmetrics import Accuracy
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import cv2

class Net(nn.Module):
        def __init__(self, drop=0.2):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5) 
        # output size = 6 *44*44 values 
        # image size : n*n 
        # filter size: f*f (f is odd number)
        # shrinked_image size : (n - f + 1)^2 

            self.bn1 = nn.BatchNorm2d(6)  # Batch normalization after conv1
            
            self.pool = nn.MaxPool2d(2, 2)
        # default stride is 2 because it was not specified so defaults to kernel size which is 2
        # output size = ((n-f+1)/2)^2 = 22*22 *6  
            
            self.conv2 = nn.Conv2d(6, 16, 5)
        #output size = 18 * 18 * 16 = 5184   
            
            self.bn2 = nn.BatchNorm2d(16)  # Batch normalization after conv2
            
            self.fc1 = nn.Linear(16 * 9 * 9, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)
            
            self.dropout = nn.Dropout(p=drop)
            
        def forward(self, x):
            x = self.pool(F.relu(self.bn1(self.conv1(x)))) 
            # 44*44*6 , 22*22*6 
            
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            # 18*18*16 , 9*9*16 
            
            x = torch.flatten(x, 1) # flatten all dimensions except batch
            x = F.relu(self.dropout(self.fc1(x)))
            # x = self.dropout(x)
            x = F.relu(self.dropout(self.fc2(x)))
            # x = self.dropout(x)
            x = self.fc3(x)
            return x
        
        
train_folder_path = './data/FER2013Train'
test_folder_path = './data/FER2013Test'
valid_folder_path = './data/FER2013Valid'


class FERPlusDataset(Dataset):
    """FERPlus dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        # Get the unique classes from the emotions column
        self.classes = self.img_frame.iloc[:, 2:].shape[1]
        
    def __len__(self):
        return len(self.img_frame)

#     to access elements using the []
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
#   to create the image name
        img_name = os.path.join(self.root_dir, self.img_frame.iloc[idx, 0])

        gray_image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        color_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        
        # image = io.imread(img_name)
        image = Image.fromarray(color_image)
        emotions = self.img_frame.iloc[idx, 2:]
        emotions = np.asarray(emotions)
        emotions = emotions.astype('float32')

        sample = {'image': image, 'emotions': emotions} # a dictionary of an image with its label
        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample #return a transformed image with label
    
        
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, emotions = sample['image'], sample['emotions']

        # Convert grayscale image to RGB
        image_rgb = np.repeat(image[..., np.newaxis], 3, axis=-1)

        transform = transforms.ToTensor()

        return {'image': transform(image_rgb),
                'emotions': emotions}


mu, st = 0, 1

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(48, scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(mu,), std=(st,))
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=(mu,), std=(st,))
    ]),
}



train_dataset = FERPlusDataset(os.path.join(train_folder_path,"label.csv"), train_folder_path, transform=data_transforms['train'])
valid_dataset = FERPlusDataset(os.path.join(valid_folder_path, "label.csv"), valid_folder_path, transform=data_transforms['val'])
test_dataset = FERPlusDataset(os.path.join(test_folder_path, "label.csv"), test_folder_path, transform=data_transforms['val'])


