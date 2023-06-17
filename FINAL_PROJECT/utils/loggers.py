import csv
import time
import matplotlib.pyplot as plt


    
def write_RESNET_details_to_table (outputs_path, epochs, optimizer ,criterion, batch_size, learning_rate, activation_func, elapsed_time,train_loss, train_accuracy, valid_loss, valid_accuracy, trial_id, scheduler, device_name, aug, weight_freezing): 
    """
    Writes the details of a RESNET model training run to a CSV file.
    
    :param outputs_path: Path to the directory where the CSV file is saved.
    :type outputs_path: str
    :param epochs: Number of epochs the model was trained for.
    :type epochs: int
    :param optimizer: The optimizer used for training.
    :type optimizer: torch.optim.Optimizer
    :param criterion: The loss function used for training.
    :type criterion: torch.nn.Module
    :param batch_size: The batch size used for training.
    :type batch_size: int
    :param learning_rate: The initial learning rate used for training.
    :type learning_rate: float
    :param activation_func: The activation function used in the model.
    :type activation_func: str
    :param elapsed_time: The time taken to complete the training.
    :type elapsed_time: float
    :param train_loss: The list of training losses during the training run.
    :type train_loss: List[float]
    :param train_accuracy: The list of training accuracies during the training run.
    :type train_accuracy: List[float]
    :param valid_loss: The list of validation losses during the training run.
    :type valid_loss: List[float]
    :param valid_accuracy: The list of validation accuracies during the training run.
    :type valid_accuracy: List[float]
    :param trial_id: The ID of the training run.
    :type trial_id: int
    :param scheduler: The learning rate scheduler used during training.
    :type scheduler: torch.optim.lr_scheduler._LRScheduler
    :param device_name: The name of the device used for training.
    :type device_name: str
    :param aug: Whether data augmentation was used during training.
    :type aug: bool
    :param weight_freezing: Whether weight freezing was used during training.
    :type weight_freezing: bool
    """
    with open(f'{outputs_path}/statistics.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        if csvfile.tell() == 0:
            # Write the column headers
            writer.writerow(['trial', 'Batch size', 'Epochs', 'Activation function', 'Loss function', 'Initial Learning rate', 'Optimizer', 'Scheduler',
                             'Min training loss', 'Average training loss', 'Max training loss', 
                             'Min training accuracy %', 'Average training accuracy %', 'Max training accuracy %',
                             'Min validation loss', 'Average validation loss', 'Max validation loss', 
                             'Min validation accuracy %', 'Average validation accuracy %',  'Max validation accuracy %', 
                             'Total time','Device', 'Data augmentation', 'Weight freezing'])

        # Write the row of data
        writer.writerow([trial_id, batch_size, epochs, activation_func, criterion, learning_rate, optimizer, scheduler,
                         round(min(train_loss), 2), round(sum(train_loss) / len(train_loss), 2), round(max(train_loss), 2),
                         round(min(train_accuracy), 2), round(sum(train_accuracy) / len(train_accuracy), 2), round(max(train_accuracy), 2),
                         round(min(valid_loss), 2), round(sum(valid_loss) / len(valid_loss), 2), round(max(valid_loss), 2),
                         round(min(valid_accuracy), 2), round(sum(valid_accuracy) / len(valid_accuracy), 2), round(max(valid_accuracy), 2),
                         time.strftime('%H:%M:%S', time.gmtime(elapsed_time)), device_name, aug, weight_freezing])
        

def write_to_table (outputs_path, epochs, optimizer ,criterion, batch_size, learning_rate, activation_func, elapsed_time,train_loss, train_accuracy, valid_loss, valid_accuracy, scheduler, device_name, dropout): 
    """
    Writes the statistics of a training run to a csv file. 

    Args:
        outputs_path (str): Path to the directory where the statistics csv file is located.
        epochs (int): Number of epochs the model was trained for.
        optimizer (torch.optim): The optimizer used in training.
        criterion (torch.nn.modules.loss): The loss function used in training.
        batch_size (int): The batch size used in training.
        learning_rate (float): The learning rate used in training.
        activation_func (str): The activation function used in training.
        elapsed_time (float): The time taken to train the model.
        train_loss (list): A list of training losses.
        train_accuracy (list): A list of training accuracies.
        valid_loss (list): A list of validation losses.
        valid_accuracy (list): A list of validation accuracies.
        trial_id (str): A unique identifier for the training run.
        scheduler (torch.optim.lr_scheduler): The learning rate scheduler used in training.
        device_name (str): The name of the device used in training.
        dropout (float): The dropout rate used in training.

    Returns:
        None
    """
    with open(f'{outputs_path}/statistics.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        if csvfile.tell() == 0:
            # Write the column headers
            writer.writerow(['Batch size', 'Epochs', 'Activation function', 'Loss function', 'Initial Learning rate', 'Optimizer', 'Scheduler', 'dropout rate',
                             'Min training loss', 'Average training loss', 'Max training loss', 
                             'Min training accuracy %', 'Average training accuracy %', 'Max training accuracy %',
                             'Min validation loss', 'Average validation loss', 'Max validation loss', 
                             'Min validation accuracy %', 'Average validation accuracy %',  'Max validation accuracy %', 
                             'Total time','Device'])

        # Write the row of data
        writer.writerow([batch_size, epochs, activation_func, criterion, learning_rate, optimizer, scheduler, dropout,
                         round(min(train_loss), 2), round(sum(train_loss) / len(train_loss), 2), round(max(train_loss), 2),
                         round(min(train_accuracy), 2), round(sum(train_accuracy) / len(train_accuracy), 2), round(max(train_accuracy), 2),
                         round(min(valid_loss), 2), round(sum(valid_loss) / len(valid_loss), 2), round(max(valid_loss), 2),
                         round(min(valid_accuracy), 2), round(sum(valid_accuracy) / len(valid_accuracy), 2), round(max(valid_accuracy), 2),
                         time.strftime('%H:%M:%S', time.gmtime(elapsed_time)), device_name])


def create_accuracy_loss_plot(outputs_path, epochs, train_accuracy, valid_accuracy, train_loss, valid_loss):      
    """
    Creates a plot of the training and validation loss and accuracy over the specified number of epochs.
    
    Args:
        outputs_path (str): The path to the directory where the plot will be saved.
        epochs (int): The total number of epochs.
        train_accuracy (list): A list of the training accuracy values for each epoch.
        valid_accuracy (list): A list of the validation accuracy values for each epoch.
        train_loss (list): A list of the training loss values for each epoch.
        valid_loss (list): A list of the validation loss values for each epoch.
        trial_id (int): The ID of the trial to be used in the filename of the saved plot.
    
    Returns:
        None.
    """
    # Plotting the loss and accuracy
    plt.figure(figsize=(10, 5))

    # Training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), train_loss, label='Training')
    plt.plot(range(1, epochs+1), valid_loss, label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), train_accuracy, label='Training')
    plt.plot(range(1, epochs+1), valid_accuracy, label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{outputs_path}/plots/acc-loss_plot.png")
    plt.close()


def output_details_to_text (outputs_path, model, epochs, optimizer, scheduler ,criterion, batch_size, learning_rate, id):    

    """
    Writes the details of a trial to a text file.

    Args:
        outputs_path (str): The path where the output file will be created.
        model: The model used in the trial.
        epochs (int): The number of epochs the model was trained for.
        optimizer: The optimizer used during training.
        scheduler: The learning rate scheduler used during training.
        criterion: The loss function used during training.
        batch_size (int): The batch size used during training.
        learning_rate (float): The initial learning rate used during training.
        id (int): The identifier of the trial.

    Returns:
        None
    """
    # Create a file to write the output
    filename = f'{outputs_path}/trial_details/trial_{id}'
    output_file = open(filename, "w")

    output_file.write(f"Parameter Combination: \n")
    output_file.write(f"epochs: {epochs} \n")
    output_file.write(f"initial learning_rate: {learning_rate} \n")
    output_file.write(f"batch_size: {batch_size} \n")
    output_file.write(f"optimizer: {optimizer} \n")
    output_file.write(f"scheduler: {scheduler} \n")
    output_file.write(f"criterion: {criterion} \n")
    output_file.write(f"\n")
    output_file.write(f"\n")
    output_file.write(f"architecture: {model}")
    
    output_file.write(f"Finished Training with this combination\n")
    
    output_file.write("#"*70)
    output_file.close()    
    
    
def output_details_to_text (path, model, epochs, optimizer, scheduler ,criterion, batch_size, learning_rate, id):    
    # Create a file to write the output
    filename = f'{path}/trial_details/trial_{id}'
    output_file = open(filename, "w")

    output_file.write(f"Parameter Combination: \n")
    output_file.write(f"epochs: {epochs} \n")
    output_file.write(f"initial learning_rate: {learning_rate} \n")
    output_file.write(f"batch_size: {batch_size} \n")
    output_file.write(f"optimizer: {optimizer} \n")
    output_file.write(f"scheduler: {scheduler} \n")
    output_file.write(f"criterion: {criterion} \n")
    output_file.write(f"\n")
    output_file.write(f"\n")
    output_file.write(f"architecture: {model}")
    
    output_file.write(f"Finished Training with this combination\n")
    
    output_file.write("#"*70)
    output_file.close()