import time
import torch.optim as optim
import data.FerPlus as FP
import torch
import torch.nn.functional as F
from utils.loggers import write_RESNET_details_to_table, write_to_table, create_accuracy_loss_plot, output_details_to_text

device = FP.get_device()
if device == torch.device("cpu"):
    device_name = 'CPU'
elif device == torch.device("mps"):
    device_name = 'MPS'
    
   
def train_and_validate(path, epochs, optimizer, scheduler ,criterion, model, trainloader, validloader, batch_size, learning_rate, activation_func = F.relu, trial_id=0):
    """
    Trains and validates the given model using the provided data loaders and hyperparameters.
    
    :param path: str - Path to the directory to save the model and plots.
    :param epochs: int - Number of epochs to train the model.
    :param optimizer: function - The optimizer function to use.
    :param scheduler: function - The learning rate scheduler function to use.
    :param criterion: function - The loss function to use.
    :param model: torch.nn.Module - The model to train and validate.
    :param trainloader: torch.utils.data.DataLoader - The data loader for the training set.
    :param validloader: torch.utils.data.DataLoader - The data loader for the validation set.
    :param batch_size: int - The batch size to use for training and validation.
    :param learning_rate: float - The learning rate to use for the optimizer.
    :param activation_func: function (optional) - The activation function to use.
    :param trial_id: int (optional) - The id to use in the file name when saving the model.
    
    :returns: tuple - A tuple containing the trained model, training and validation accuracies and losses, time elapsed during training, and the learning rate scheduler used.
    """
    train_loss = []
    train_accuracy = []
    valid_loss = []
    valid_accuracy = []
    opt_name = optimizer.__name__
    optimizer = optimizer(model.parameters(), lr=learning_rate)
    
    if scheduler == optim.lr_scheduler.ReduceLROnPlateau:
        scheduler = scheduler(optimizer)
        # print("plateu")
        # print(type(scheduler))
        
    elif scheduler == optim.lr_scheduler.ExponentialLR: 
        scheduler = scheduler(optimizer, gamma=0.9)
        # print(type(scheduler))
        
    st = time.time()

# Training - Validation loop
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        model.train()
        
        # Perform training
        for data in trainloader:
            labels = data['emotions'].to(device)
            inputs = data['image'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
            # Calculate and store training accuracy
            _, predicted = torch.max(outputs, 1)
            _, labels = torch.max(labels, 1)
            
            total += inputs.size(0)
            correct +=  (predicted == labels).sum().item()
            
        scheduler.step()
        train_loss.append(running_loss / len(trainloader))
        train_accuracy.append(100 * correct / total)
        
        # Perform validation
        model.eval()
        correct = 0
        total = 0
        running_loss = 0.0
        with torch.inference_mode():
            for data in validloader:
                labels = data['emotions'].to(device)
                inputs = data['image'].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                _, labels = torch.max(labels, 1)
                total += inputs.size(0)
                correct +=  (predicted == labels).sum().item() 
        
        valid_loss.append(running_loss / len(validloader))
        valid_accuracy.append(100 * correct / total)
        
        # Print the training and validation loss and accuracy
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Training Loss: {train_loss[-1]:.4f} | Training Accuracy: {train_accuracy[-1]:.2f}%')
        print(f'Validation Loss: {valid_loss[-1]:.4f} | Validation Accuracy: {valid_accuracy[-1]:.2f}%')
        print('-----------------------------------')

    elapsed_time = time.time() - st
    print('Finished Training')
    
    # torch.save(model.state_dict(), f'./models/RESNET/RESNET-18_{trial_id}.pth')     
    
    return model, train_accuracy, train_loss, valid_accuracy, valid_loss, elapsed_time, scheduler
    write_to_table(path, epochs, opt_name, criterion, batch_size, learning_rate, activation_func.__name__, elapsed_time, train_loss, train_accuracy, valid_loss, valid_accuracy, trial_id, scheduler.__class__.__name__, device_name, dropout)
    
    create_accuracy_loss_plot(path, epochs, train_accuracy, valid_accuracy, train_loss, valid_loss)
    