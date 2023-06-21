from test import test_model
from train_and_validate import train_and_validate
from utils.loggers import output_details_to_text, write_RESNET_details_to_table, write_to_table, create_accuracy_loss_plot
from utils.hyper_paramaters import hps
from models.CNN import Net 
from data.FerPlus import FERPlusDataset, get_classes, get_data_loaders, get_data_transforms, get_device, get_datasets
from utils.dir import create_output_directories
import torch
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from data.FerPlus import FERPlusDataset
import torch.nn as nn
import matplotlib.pyplot as plt


def train_and_test_untrained_CNN():
    """
    Runs a complete training and testing process for an untrained neural network model.
    hyper params are taken from "FINAL_PROJECT/utils/hyper_parameters.py" file, if you want to change any parameters change it there.
    saves training and validation details in csv file
    create accuracy-loss plot for training and validation
    creates confusion matrix for testing
    Parameters:
    -----------
    None
    
    Returns:
    --------
    None
    """
    criterion, optimizer, activation_func, learning_rate, epochs, batch_size, scheduler, dropout_rate = hps['CNN'].values()
    opt_name = optimizer.__name__
    device = get_device()
    trainloader, validloader, testloader = get_data_loaders()
    model = Net(drop=dropout_rate, activation_func=activation_func)
    model = model.to(device)
    optimizer = optimizer(model.parameters(), lr = learning_rate)
    path = create_output_directories('outputs')
    model, train_accuracy, train_loss, valid_accuracy, valid_loss, elapsed_time, scheduler_name = train_and_validate(path=path, epochs=epochs, optimizer=optimizer, scheduler=scheduler, criterion=criterion, model=model, trainloader=trainloader, validloader=validloader, batch_size=batch_size, learning_rate=learning_rate, activation_func=activation_func)
    
    test_model(model, testloader, path)
    
    write_to_table(path, epochs, opt_name, criterion, batch_size, learning_rate, activation_func.__name__, elapsed_time, train_loss, train_accuracy, valid_loss, valid_accuracy, scheduler_name.__class__.__name__, device, dropout_rate)
    
    create_accuracy_loss_plot(path, epochs, train_accuracy, valid_accuracy, train_loss, valid_loss)


def load_and_test_pretrained_CNN(pretrained_model_path):
    """
    Loads a pretrained Convolutional Neural Network model from a specified path
    and evaluates the model on the test dataset. It creates a confusion matrix.
    This function does not train the
    model. 
    
    :param pretrained_model_path: A string specifying the path to the pretrained model.
    
    :return: None.
    """
    trainloader, validloader, testloader = get_data_loaders()
    model = Net()
    model.load_state_dict(torch.load(pretrained_model_path))
    print("model loaded successfully")
    model.eval()
    output_path = create_output_directories(pretrained_model_path)
    
    test_model(model, testloader, output_path, pretrained_model_path)


def train_and_test_pretrained_pretrained_RESNET(weight_freezing=False):
    """
    Trains,validates and test a pre-trained ResNet-18 model on a custom dataset. By default, all weights of the
    model are trainable, but by setting `weight_freezing` to True, all layers except for the first
    and last ones will have their weights frozen. 

    Args:
    - weight_freezing (bool): Whether to freeze all layers except the first and last ones. Default is False.

    Returns:
    - None
    """
    criterion, optimizer, activation_func, learning_rate, epochs, batch_size, scheduler, dropout_rate = hps['RESNET'].values()
    opt_name = optimizer.__name__
    device = get_device()
    trainloader, validloader, testloader = get_data_loaders(augmentation=False)
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    if weight_freezing:
        print("Weights of all layers in the model exept of first and last layers will be freezed")
        for param in model.parameters():
            param.requires_grad = False
    else:
        print("all weights of all layers of the models are trainable .. this might take longer")        
    train_dataset, validation_dataset, test_dataset = get_datasets(augmentation=False)
    
    num_classes = train_dataset.classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    print("model loaded successfully")
    model.to(device)
    path = create_output_directories()
    # model.load_state_dict(torch.load('../models/RESNET/RESNET-18_11.pth'))
    # print(f"model: {model}")
    model, train_accuracy, train_loss, valid_accuracy, valid_loss, elapsed_time, scheduler_name = train_and_validate(path=path, epochs=epochs, optimizer=optimizer, scheduler=scheduler, criterion=criterion, model=model, trainloader=trainloader, validloader=validloader, batch_size=batch_size, learning_rate=learning_rate)
    
    test_model(model, testloader, path)
    
    write_to_table(path, epochs, opt_name, criterion, batch_size, learning_rate, activation_func.__name__, elapsed_time, train_loss, train_accuracy, valid_loss, valid_accuracy, scheduler_name.__class__.__name__, device, dropout_rate)
    
    create_accuracy_loss_plot(path, epochs, train_accuracy, valid_accuracy, train_loss, valid_loss)


def load_and_test_RESNET(pretrained_model_path):
    """
    Load a pre-trained RESNET model, perform testing, and save the results.

    Args:
        pretrained_model_path (str): Path to the pre-trained RESNET model file.

    Returns:
        None

    """
    criterion, optimizer, activation_func, learning_rate, epochs, batch_size, scheduler, dropout_rate = hps['RESNET'].values()
    opt_name = optimizer.__name__
    device = get_device()
    trainloader, validloader, testloader = get_data_loaders(augmentation=False)
    train_dataset, validation_dataset, test_dataset = get_datasets(augmentation=False)
    
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    num_classes = train_dataset.classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # load model from path
    model.load_state_dict(torch.load(pretrained_model_path))    
       
    path = create_output_directories(pretrained_model_path=pretrained_model_path)
    
    test_model(model, testloader, path, pretrained_model_path)
    