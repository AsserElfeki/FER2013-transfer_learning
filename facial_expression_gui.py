import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import ImageTk, Image
import torch
from FE_model import Net
from torchvision.transforms import ToTensor
global model 
selected_option = ''
classes = {
    0: 'Neutral',
    1: 'Happinnes',
    2: 'Surprise',
    3: 'Sadness',
    4: 'Anger',
    5: 'Disgust',
    6: 'Fear',
    7: 'Contempt',
    8: 'Unknown',
    9: 'NF'
}

def create_data_loader (img):
    # Convert the preprocessed image data to a PyTorch tensor
    tensor_data = ToTensor()(img)

    # Add an extra dimension to represent the batch size of 1
    tensor_data = tensor_data.unsqueeze(0)
    # Create a DataLoader with the single image tensor
    dataloader = torch.utils.data.DataLoader(tensor_data, batch_size=1, shuffle=False)

    return dataloader

        
def classify_image(dataloader):
    predicted_class = ''
    
    for data in dataloader: 
        preprocessed_image = data
        output = model(preprocessed_image)
        
        _, predicted_index = torch.max(output, 1)
        predicted_class = classes[predicted_index.item()]
    
    return predicted_class    
    

def choose_image():
    print(model)
    file_path = filedialog.askopenfilename(filetypes=[("JPEG Files", "*.jpeg"), ("PNG Files", "*.png"), ("JPG files", "*.jpg"), ("WEBP file", "*.webp")]
)
    if file_path:
        # Perform face detection and cropping using cv2
        image = cv2.imread(file_path)
        # show the image
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(
            image, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cropped_img = image[y:y+h, x:x+w]
            resized_img = cv2.resize(cropped_img, (48, 48))
            gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
            # pil_image = Image.fromarray(gray_img)
            # resized_pil_image = pil_image.resize((200, 200))
            # image_tk = ImageTk.PhotoImage(resized_pil_image)
            # images.push(image_tk)
            if selected_option == 'RESNET':
                label = classify_image(create_data_loader(resized_img))
            elif selected_option == 'CNN':
                label = classify_image(create_data_loader(gray_img))
            cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12) ,2)        

        cv2.imshow('original', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
def exit_program():
    window.destroy()
    
    
def capture_photo():
    
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
    
        for (x,y,w,h) in faces:
            # To draw a rectangle in a face 
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),2) 
            cropped_img = image[y:y+h, x:x+w]
            resized_img = cv2.resize(cropped_img, (48, 48))
            gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
            if selected_option == 'RESNET':
                label = classify_image(create_data_loader(resized_img))
            elif selected_option == 'CNN':
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
    global model
    print("choose RESNET")
      
    model = models.resnet18(pretrained=True, progress=True)

    num_classes = len(FE_model.train_dataset.classes)
    in_features = model.fc.in_features

    # Modify the classifier
    model.fc = FE_model.nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load('./pretrained_resnet18.pt'))
    model.eval()
    print(model)
def choose_my_CNN():
    global model
    print("choose CNN")
    model = FE_model.Net()
    model.load_state_dict(torch.load('./mymodel.pth'))
    model.eval()    
    print(model)

def choose_mode(option):
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


# Add buttons
choose_button = tk.Button(window, text="Choose Image", command=choose_image, bg=bg_color)
choose_button.pack()


capture_button = tk.Button(window, text="Take Photo", command=capture_photo)
capture_button.pack()


selected_option = tk.StringVar(window)
selected_option.set("RESNET18")  # Set default option

mode_button = tk.OptionMenu(window, selected_option, "RESNET18", "CNN", command=choose_mode)
mode_button.pack()


# exit button at the bottom right corner
exit_button = tk.Button(window, text="Exit", command=exit_program)
# exit_button.pack()



window.mainloop()
