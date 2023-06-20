from loops import train_and_test_pretrained_pretrained_RESNET, train_and_test_untrained_CNN, load_and_test_pretrained_CNN, load_and_test_RESNET
import os

# load_and_test_RESNET('../models/RESNET/RESNET-18_11.pth')
# load_and_test_pretrained_CNN('../models/myCNN_noAug_30_epochs.pth')

# 1 to 16
# for file_name in os.listdir('../stats/RESNET-Final/models'):
#         if os.path.isfile(os.path.join('../stats/RESNET-Final/models', file_name)):
#             load_and_test_RESNET(os.path.join('../stats/RESNET-Final/models', file_name))

for file_name in os.listdir('../stats/CNNs-Final/models'):
        if os.path.isfile(os.path.join('../stats/CNNs-Final/models', file_name)):
            load_and_test_pretrained_CNN(os.path.join('../stats/CNNs-Final/models', file_name))
            

for file_name in os.listdir('../stats/CNNs-2/models'):
        if os.path.isfile(os.path.join('../stats/CNNs-2/models', file_name)):
            load_and_test_pretrained_CNN(os.path.join('../stats/CNNs-2/models', file_name))            

# load_and_test_pretrained_CNN('../stats/CNNs-Final/models/trial_1.pth')

# load_and_test_RESNET('../stats/RESNET-Final/models/trial_15.pth')
