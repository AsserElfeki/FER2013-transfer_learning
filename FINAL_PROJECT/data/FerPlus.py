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
        Initializes a new instance of the FERPlusDataset class.

        Parameters:
        -----------
        csv_file : str
            The path to the CSV file with annotations describing the dataset.
        root_dir : str
            The directory containing all the images of the dataset.
        transform : callable, optional
            An optional transform to be applied to a sample.
        """
        self.img_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        # Get the unique classes from the emotions column
        self.classes = self.img_frame.iloc[:, 2:].shape[1]

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
        --------
        int
            The total number of samples in the dataset.
        """
        return len(self.img_frame)

    def __getitem__(self, idx):
        """
        Retrieves a sample of transformed image with label from the dataset.

        Parameters:
        -----------
        idx : int or float
            Index of the image data to be retrieved.

        Returns:
        --------
        dict
            A dictionary containing an image with transformed pixel values and its corresponding
            label containing the emotions expressed in the image.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Create the image name
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

def get_data_transforms(augmentation=True):
    """
    Returns a dictionary containing PyTorch transforms for the input data, with
    two keys: 'train' and 'valid'. If 'augmentation' is True, applies a series
    of random augmentations to the 'train' transforms, including random affine
    transformations, random horizontal flips, and random rotations up to 10
    degrees. All transforms include conversion to tensor and normalization
    using mean=[0.485] and std=[0.229]. If 'augmentation' is False, returns
    the same transforms without any augmentations applied.
    
    Parameters:
    -----------
    augmentation : bool, optional (default=True)
        If True, apply random augmentations to the 'train' transforms.
        
    Returns:
    --------
    data_transforms : dict
        A dictionary containing PyTorch transforms for the input data, with two
        keys: 'train' and 'valid'. Each key maps to a Compose object containing
        a sequence of transforms to be applied to the data.
    """
    torch.manual_seed(17) #https://pytorch.org/vision/stable/transforms.html
    
    if augmentation:
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomApply([transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229]),
            ]),
            'valid': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229]),
            ]),
        }
        
    else: 
        data_transforms = {
            'train': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229]),
            ]),
            'valid': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229]),
            ]),
        }
    return data_transforms        

def get_datasets(train_folder_path = '../data/FER2013Train', test_folder_path = '../data/FER2013Test', valid_folder_path = '../data/FER2013Valid', augmentation=True):
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
    data_transforms = get_data_transforms(augmentation=augmentation)
    
    train_dataset = FERPlusDataset(csv_file= os.path.join(train_folder_path,"label.csv"), root_dir=train_folder_path, transform=data_transforms['train'])
    validation_dataset = FERPlusDataset(csv_file= os.path.join(valid_folder_path, "label.csv"), root_dir= valid_folder_path, transform=data_transforms['valid'])
    test_dataset = FERPlusDataset(csv_file= os.path.join(test_folder_path, "label.csv"), root_dir= test_folder_path, transform=data_transforms['valid'])
    
    return train_dataset, validation_dataset, test_dataset
    
    
def get_data_loaders(batch_size=hps['CNN']['batch_size'], augmentation=True):
    """
    Get data loaders for training, validation, and testing datasets.

    Args:
        batch_size (int, optional): Batch size for the data loaders. Default is the value specified in hps['CNN']['batch_size'].
        augmentation (bool, optional): Flag to enable data augmentation. Default is True.

    Returns:
        torch.utils.data.DataLoader: Data loader for the training dataset.
        torch.utils.data.DataLoader: Data loader for the validation dataset.
        torch.utils.data.DataLoader: Data loader for the testing dataset.

    """
    train_dataset, validation_dataset, test_dataset = get_datasets(augmentation=augmentation)
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