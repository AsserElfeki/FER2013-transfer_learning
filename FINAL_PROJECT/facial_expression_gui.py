import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import ImageTk, Image
import torch
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import ResNet18_Weights
import torch.nn as nn
from data.FerPlus import get_classes, get_data_loaders, get_datasets
from models.CNN import Net

global model 
selected_option = ''
classes = get_classes(mode='validate')

def create_data_loader (img):
    """
    Creates a PyTorch DataLoader from a preprocessed image tensor.

    Args:
        img: A preprocessed image tensor.

    Returns:
        A PyTorch DataLoader object with a batch size of 1 containing the input image tensor.
    """
    # Convert the preprocessed image data to a PyTorch tensor
    tensor_data = ToTensor()(img)

    # Add an extra dimension to represent the batch size of 1
    tensor_data = tensor_data.unsqueeze(0)
    tensor_data = transforms.Normalize(mean=[0.485], std=[0.229]) (tensor_data)
    # Create a DataLoader with the single image tensor
    dataloader = torch.utils.data.DataLoader(tensor_data, batch_size=1, shuffle=False)

    return dataloader

def classify_image(dataloader):
    """
    Given a dataloader, this function classifies the images in it using a pre-trained model. 
    The function returns the predicted class of the image.
    
    Parameters:
    dataloader (DataLoader): The dataloader containing the images to be classified.
    
    Returns:
    predicted_class (str): The predicted class of the image.
    """
    predicted_class = ''
    
    for data in dataloader: 
        preprocessed_image = data
        output = model(preprocessed_image)
        
        _, predicted_index = torch.max(output, 1)
        predicted_class = classes[predicted_index.item()]
    
    return predicted_class    

def choose_image():
    """
    Allows the user to choose an image file from their file system, perform face detection and cropping using the OpenCV library, and display the resulting image with a label of the detected object. Returns nothing.

    Parameters:
    None

    Returns:
    None
    """
    file_path = filedialog.askopenfilename(filetypes=[("JPEG Files", "*.jpeg"), ("PNG Files", "*.png"), ("JPG files", "*.jpg"), ("WEBP file", "*.webp")]
)
    if file_path:
        # Perform face detection and cropping using cv2
        image = cv2.imread(file_path)
        # show the image
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(
            image, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        if len(faces) == 0:
            resized_img = cv2.resize(image, (48, 48))
            gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
            print("reached here")
            label = classify_image(create_data_loader(gray_img))
            
            cv2.putText(image, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12) ,2)
            # Display an image in a window
        else:    
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cropped_img = image[y:y+h, x:x+w]
                resized_img = cv2.resize(cropped_img, (48, 48))
                gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
                
                label = classify_image(create_data_loader(gray_img))
                cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12) ,2)        

        cv2.imshow('original', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def exit_program():
    """
    Stops the program by destroying the window.
    Does not have any parameters.
    Does not return anything.
    """
    window.destroy()

def capture_photo():
    """
    Capture photo from the camera and perform face detection and classification.

    Returns:
        None

    """
    cap = cv2.VideoCapture(0)
  
    # loop runs if capturing has been initialized.
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    while 1:  
        # reads frames from a camera
        ret, image = cap.read() 
    
        # convert to gray scale of each frames
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detects faces of different sizes in the input image
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            resized_img = cv2.resize(image, (48, 48))
            gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
            
            label = classify_image(create_data_loader(gray_img))
            
            cv2.putText(image, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12) ,2)
            # Display an image in a window
            
        else:        
            for (x,y,w,h) in faces:
                # To draw a rectangle in a face 
                cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),2) 
                cropped_img = image[y:y+h, x:x+w]
                resized_img = cv2.resize(cropped_img, (48, 48))
                gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
                
                label = classify_image(create_data_loader(gray_img))
                        
                cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12) ,2)
            # Display an image in a window
        cv2.imshow('img',image)
        
        # Wait for Esc key to stop
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    
    # Close the window
    cap.release()
    
    # De-allocate any associated memory usage
    cv2.destroyAllWindows() 

def choose_transfer_learning():  
    """
    Initializes a ResNet18 model for transfer learning, modifies its classifier for the number of classes in the dataset,
    loads the pre-trained weights obtained after 10 epochs of fine-tuning, and sets the model in evaluation mode.
    This function takes no parameters and returns nothing.
    """
    global model
    print("choose RESNET")
    train_dataset, validation_dataset, test_dataset = get_datasets(augmentation=False)
      
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    num_classes = train_dataset.classes
    in_features = model.fc.in_features

    # Modify the classifier
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(in_features=in_features, out_features=num_classes)
    # model.load_state_dict(torch.load('../models/RESNET/RESNET-18_11.pth'))
    model.load_state_dict(torch.load('../stats/RESNET-Final/models/trial_3.pth'))
    # model.load_state_dict(torch.load('../stats/RESNET-Final/models/trial_11.pth'))
    
    
    # model.load_state_dict(torch.load('../models/pretrained_resnet18_10_epochs.pt'))
    
    model.eval()
    
def choose_my_CNN():
    """
    Initializes a global PyTorch model and loads the weights from a pre-trained state dictionary. 
    This function takes no parameters and has no return types.
    """
    global model
    print("choose CNN")
    model = Net()
    
    # model.load_state_dict(torch.load('../models/mymodel.pth'))
    
    # model.load_state_dict(torch.load('../stats/outputs-7-no_aug/trial_1.pth'))
    model.load_state_dict(torch.load('../stats/CNNs-2/models/trial_2.pth'))
    # model.load_state_dict(torch.load('../stats/CNNs/models/trial_25.pth'))
    
    
    
    model.eval()    

def choose_mode(option):
    """
    Chooses a machine learning model based on the selected option.

    Args:
        option (str): The selected machine learning model option. Possible values are "RESNET18" and "CNN".
    
    Returns:
        None
    """
    global model
    global selected_option
    
    if option == "RESNET18":
        choose_transfer_learning()
        selected_option = 'RESNET'
    elif option == "CNN":
        choose_my_CNN()
        selected_option = 'CNN'

        
        
window = tk.Tk()
window.title("Smaluch - Moustafa")
bg_color = 'light blue'
# Determine the screen resolution
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

# Calculate the window position
window_width = 500  # Replace with your desired window width
window_height = 300  # Replace with your desired window height

x = (screen_width // 2) - (window_width // 2)
y = (screen_height // 2) - (window_height // 2)

# Set the window position
window.geometry(f"{window_width}x{window_height}+{x}+{y}")

# Add labels
label = tk.Label(window, text="Facial Expression Recognition", font=("Helvetica", 16))
label.pack()

selected_option = tk.StringVar(window)
selected_option.set("select NN")  # Set default option

mode_button = tk.OptionMenu(window, selected_option, "RESNET18", "CNN", command=choose_mode)
mode_button.pack()

# Add buttons
choose_button = tk.Button(window, text="Choose Image", command=choose_image, bg=bg_color)
choose_button.pack()

capture_button = tk.Button(window, text="Live Classification", command=capture_photo)
capture_button.pack()

# exit button at the bottom right corner
exit_button = tk.Button(window, text="Exit", command=exit_program)
# exit_button.pack()

window.mainloop()