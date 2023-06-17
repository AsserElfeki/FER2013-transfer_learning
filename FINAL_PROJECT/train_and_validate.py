import time
import torch.optim as optim
import data.FerPlus as FP
import torch
from utils.loggers import write_RESNET_details_to_table, write_to_table, create_accuracy_loss_plot, output_details_to_text

device = FP.get_device()
if device == torch.device("cpu"):
    device_name = 'CPU'
elif device == torch.device("mps"):
    device_name = 'MPS'
    
   
def train_and_validate(path, epochs, optimizer, scheduler ,criterion, model, trainloader, validloader, batch_size, learning_rate, activation_func, trial_id=0):
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
    