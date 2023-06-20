import re
import string
import torch.nn.functional as F
import torch
import data.FerPlus as FP
from torchmetrics import Accuracy
from sklearn.metrics import precision_score, f1_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

device = FP.get_device()

torchmetrics_accuracy = Accuracy(task='multiclass', num_classes=10).to(device)

classes = FP.get_classes('test')

def test_model(model, test_loader, path, pretrained_model_path):
    """
    Evaluate a PyTorch model on a given test dataset and log evaluation metrics.
    
    Args:
        model (nn.Module): A PyTorch model to evaluate.
        test_loader (DataLoader): A PyTorch DataLoader for the test dataset.
        path (str): The path to save the confusion matrix plot.
        
    Returns:
        None
    """
    model.to('cpu')
    model.eval()
    correct = 0
    total = 0
    test_accuracy = []
    predicted_labels=[]
    true_labels = []
    
    wrong = 0
    with torch.inference_mode():
        for i, data in enumerate(test_loader):
            labels = data['emotions']
            inputs = data['image']
            output = model(inputs)
            
            # measure accuracy
            _, predicted = torch.max(output, 1)
            _, labels = torch.max(labels, 1)
            # torchmetrics_accuracy(predicted, labels)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_accuracy.append(torchmetrics_accuracy(predicted, labels))
            
            # Store predicted and true labels for calculating metrics
            predicted_labels.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            
            # Calculate evaluation metrics
        f1 = f1_score(true_labels, predicted_labels, average='weighted')
        precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=1)
        recall = recall_score(true_labels, predicted_labels, average='weighted')
        # confusion_mat = confusion_matrix(true_labels, predicted_labels)
        
        cm_display = ConfusionMatrixDisplay.from_predictions(y_true=true_labels, y_pred=predicted_labels, normalize='true', display_labels=classes, cmap='Blues', values_format='0.2f', xticks_rotation=45)
        fig, ax = plt.subplots(figsize=(10,8))
        cm_display.plot(ax=ax)
        id = re.findall(r'_(\d+)',pretrained_model_path)
        id = int(id[0])
        
        plot_path = os.path.join(path, f'CM-{id}.png')
        plt.savefig(plot_path)
        with open(os.path.join(path,f'output-{id}.txt'), 'w') as file:
            # Log evaluation metrics
            file.write(f"{pretrained_model_path}\n\n")
            file.write("Evaluation Metrics:\n")
            file.write(f"Accuracy of the network on the test images: {100 * (sum(test_accuracy) / len(test_accuracy)):.2f}%\n")
            file.write(f"F1 Score: { f1:.2f}\n")
            file.write(f"Precision: { precision:.2f}\n")
            file.write(f"Recall: { recall:.2f}\n")
            file.write(f"correct: {correct}, total: {total}\n")
        
        # # Log evaluation metrics
        # print("Evaluation Metrics:")
        # print(f"average accuracy : {100* (sum(test_accuracy) / len(test_accuracy)):.2f}%")
        # print(f"F1 Score: {100* f1 :.2f}%")
        # print(f"Precision: {100* precision :.2f}%")
        # print(f"Recall: {100* recall :.2f}%")
        # print("Confusion Matrix:")
        # # print(confusion_mat)
        
        # # print total accuracy
        # print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
        # print(f"correct: {correct}, total: {total}")
        # # print(test_accuracy)
        # print(sum(test_accuracy).item() / len(test_accuracy))
