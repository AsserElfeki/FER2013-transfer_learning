from loops import train_and_test_pretrained_pretrained_RESNET, train_and_test_untrained_CNN, load_and_test_pretrained_CNN, load_and_test_RESNET
import os


cnn_paths = ['../stats/CNNs/models', '../stats/CNNs-4/models', '../models/CNN']
resnet_paths = ['../models/RESNET']

for path in cnn_paths:
    for file_name in os.listdir(path):
        if os.path.isfile(os.path.join(path, file_name)):
            load_and_test_pretrained_CNN(os.path.join(path, file_name))
            
for path in resnet_paths:
    for file_name in os.listdir(path):
        if os.path.isfile(os.path.join(path, file_name)):
            load_and_test_RESNET(os.path.join(path, file_name))            