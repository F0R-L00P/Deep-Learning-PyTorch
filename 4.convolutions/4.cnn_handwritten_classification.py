import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

from cnn_mnist_classification import CNN

#########################
# class CNN(nn.Module):
#     def __init__(self):
#         # build constructor-parent class nn.Module
#         super(CNN, self).__init__()  
#         # conv layer with batch norm and ReLU activation
#         self.conv1 = nn.Sequential(  
#             # Convolutional layer with 1 input channel, 8 output channels, and 3x3 kernel size
#             # if image was RBG input would be 3
#             # pad image to keep original size calculated as
#                 #(kernal_size -1)\2
#             #[(input_size - filter_size + 2(padding) / stride) +1] --> 
#             # [(28-3+2(1)/1)+1] = 28 (padding type is same)
#             # input size = output size
#             nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1), 
#             # Batch normalization layer for 8 channels
#             nn.BatchNorm2d(8),
#             # ReLU activation layer
#             nn.ReLU(),  
#         )

#         # max pooling layer with 2x2 kernel size
#         # output sizeof image should decrease after maxpooling
#         # (input_size - kernel_size)/stride) + 1 -->
#         # (28 - 2) / 2) + 1 = 14
#         self.pool1 = nn.MaxPool2d(kernel_size=2)  
#         # conv layer with batch norm and ReLU activation
#         self.conv2 = nn.Sequential(  
#             # Convolutional layer with 8 input channels, 32 output channels, and 5x5 kernel
#             # image output size output_size = ((28 - 5 + 24) / 1) + 1
#             # 28
#             nn.Conv2d(in_channels=8, out_channels=32, kernel_size=5, padding=2),  
#             # Batch normalization layer for 32 channels
#             nn.BatchNorm2d(32),
#             # ReLU activation layer
#             nn.ReLU(),  
#         )

#         # Define the second max pooling layer with 2x2 kernel size
#         # second maxpool will reduce image size to 7x7
#         # (14 - 2)/2 + 1 = 7
#         self.pool2 = nn.MaxPool2d(kernel_size=2)
#         # fully connected layer input size of 32*5*5 and output size of 600
#         self.fc1 = nn.Linear(in_features=(32 * 7 * 7), 
#                              out_features=600)
#         self.dropout = nn.Dropout(p=0.5)
#         # fully connected layer  600 * 5 * 5 input - 10 output
#         self.fc2 = nn.Linear(in_features=600, 
#                              out_features=10)  

#     def forward(self, x):
#         x = self.conv1(x)  # conv layer w batch norm and ReLU activation
#         x = self.pool1(x)  # max pooling layer
#         x = self.conv2(x)  # conv layer w batch norm and ReLU activation
#         x = self.pool2(x)  # max pooling layer
#         # Flatten the output of the second max pooling layer to a 1D tensor
#         # can use explicit value using btach_size vs -1
#         x = x.view(-1, 32 * 7 * 7)  
#         x = self.fc1(x)  # fully connected layer
#         x = self.dropout(x)
#         x = self.fc2(x)  # fully connected layer
#         return x


# load the saved model
model = CNN()
model.load_state_dict(torch.load('cnn_model_weights.pth'))
model.eval()


# Load the image and apply transformations
image_path = r'C:\Users\behna\OneDrive\Documents\GitHub\Pytorch\4.convolutions\image_files\2.png'
image = Image.open(image_path).convert('L')  # convert to grayscale
image = Image.eval(image, lambda x: 255 - x) # make background and writting colour same as training set
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # resize to (28, 28)
    transforms.ToTensor(),  # convert to tensor
    transforms.Normalize((0.1307,), (0.3081,))  # normalize
])
image_tensor = transform(image).unsqueeze(0)  # add batch dimension

# Make predictions
model.eval()
with torch.no_grad():
    output = model(image_tensor)
    _, predicted = torch.max(output, 1)

# Print the predicted digit
print('The predicted digit is:', predicted.item())