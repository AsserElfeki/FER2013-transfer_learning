from torch.utils.data import Dataset
from skimage import io
from PIL import Image
import numpy as np
import torch
import os
import pandas as pd
from torchvision import transforms, utils
from utils.hyper_paramaters import hps

class FERPlusDataset(Dataset):

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

        image = io.imread(img_name)
        # image = io.imread(image)
        image = Image.fromarray(image)

        emotions = self.img_frame.iloc[idx, 2:]
        emotions = np.asarray(emotions)
        emotions = emotions.astype('float32')

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'emotions': emotions} # a dictionary of an image with its label
        
        return sample #return a transformed image with label

def get_device ():
    """
    Returns the device available for PyTorch computation. 
    If the PyTorch Multi-Process Service (MPS) is built, the device is set to 'mps'.
    If CUDA is available, the device is set to 'cuda'.
    If no GPU is present, the device is set to 'cpu'.
    
    Returns:
    device(torch.device): The device to be used for PyTorch computation.
    """
    if torch.backends.mps.is_built():
        device = torch.device('mps')    
        
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        
    else:
        device = torch.device("cpu")
    
    return device

def get_data_transforms():
    """
    Returns a dictionary containing two Compose objects for data augmentation and normalization of
    training and validation datasets.

    Returns:
        data_transforms (dict): A dictionary containing two Compose objects for training and
        validation datasets.
    """
    data_transforms = {
            'train': transforms.Compose([
                # transforms.Grayscale(3) ,
                transforms.RandomApply([transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229]),
            ]),
            'valid': transforms.Compose([
                # transforms.Grayscale(3) ,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229]),
            ]),
        }
    
    return data_transforms        

def get_datasets(data_transforms= get_data_transforms(), train_folder_path = '../data/FER2013Train',            test_folder_path = '../data/FER2013Test', valid_folder_path = '../data/FER2013Valid'):
    """
    Returns three datasets for training, validation, and testing facial expressions recognition models.
    
    Args:
        data_transforms (dict): A dictionary of PyTorch transforms for transforming the input data.
        train_folder_path (str): Path to the folder containing the training data.
        test_folder_path (str): Path to the folder containing the testing data.
        valid_folder_path (str): Path to the folder containing the validation data.
    
    Returns:
        tuple: A tuple of three FERPlusDataset objects representing the training, validation, and testing datasets.
    """
    train_dataset = FERPlusDataset(csv_file= os.path.join(train_folder_path,"label.csv"), root_dir=train_folder_path, transform=data_transforms['train'])
    validation_dataset = FERPlusDataset(csv_file= os.path.join(valid_folder_path, "label.csv"), root_dir= valid_folder_path, transform=data_transforms['valid'])
    test_dataset = FERPlusDataset(csv_file= os.path.join(test_folder_path, "label.csv"), root_dir= test_folder_path, transform=data_transforms['valid'])
    
    return train_dataset, validation_dataset, test_dataset
    
    
def get_data_loaders( batch_size=hps['batch_size']):
    
    train_dataset, validation_dataset, test_dataset = get_datasets()
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    validloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return trainloader, validloader, testloader

def get_classes(mode):
    """
    Given a mode, this function returns a list of classes.

    :param mode: A string representing the mode of the function. 
                 Must be 'test' or 'validate'.
    :type mode: str

    :return: A list of strings representing the classes.
             If mode is 'test', the list does not include 'NF'.
             If mode is 'validate', the list includes 'NF'.
    :rtype: list
    """
    classes = [
    'Neutral',
    'Happinnes',
    'Surprise',
    'Sadness',
    'Anger',
    'Disgust',
    'Fear',
    'Contempt',
    'Unknown', 
]
    if mode == 'test':
        return classes
    elif mode == "validate":
        classes.append("NF")
        return classes