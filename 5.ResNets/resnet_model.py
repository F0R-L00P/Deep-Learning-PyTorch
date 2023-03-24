import torch
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


# Define the path where the dataset will be stored
data_path = r'C:\Users\OneDrive\Documents\GitHub\Pytorch\5.ResNets\CIFAR10_dataset'

# transformation of data
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()                 # convert images to tensors
                                ])

# Download and load the CIFAR-10
train_dataset = datasets.CIFAR10(
                                        root=data_path, 
                                        train=True, 
                                        download=True, 
                                        transform=transform
                                    )

test_dataset = datasets.CIFAR10(
                                        root=data_path, 
                                        train=False, 
                                        download=True, 
                                        transform=transform
                                  )

# check data shape
# 32x32 image for 50k train
train_dataset.data.shape
# 32`x32 image for 10k test
test_dataset.data.shape

# visualize data 
# Select a random image from the TRAIN - dataset
image, label = train_dataset[7]

# must convert 2D tensor (1, 28, 28) colour_channel, dim, dim
# to 28x28 image removing colour
image = image.numpy()[0]
# Display the image and label
plt.imshow(image)
plt.title(f"Label: {label}")
plt.axis("off")
plt.show()

# check corresponding label 
print(f'Label: {label}')


# setup dataloader for training process
# Create the DataLoader with batch size 64, shuffling
batch_size = 100
train_loader = torch.utils.data.DataLoader(train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True
                                           )
test_loader = torch.utils.data.DataLoader(test_dataset,
                                           batch_size=batch_size,
                                           shuffle=False
                                           )


# lets visualize the loader
# as batch is 100, instances are expected to be 50000\100 & 10000\100
print(f'Train Loader: {len(train_loader)} | Test Loader: {len(test_loader)}')

###################################################################
###################################################################
###################################################################

# 3x3 convolution layer
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

class ResidualBlock(nn.Module):
    # Initialize the ResidualBlock class
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        # Call the parent class constructor (nn.Module)
        super(ResidualBlock, self).__init__()
        # First convolutional layer with custom kernel size, stride, and padding
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        # Batch normalization layer after the first convolution 
        self.bn1 = nn.BatchNorm2d(out_channels)
        # ReLU activation function used in the block
        self.relu = nn.ReLU(inplace=True)
        # Second convolutional layer with custom kernel size and padding
        self.conv2 = conv3x3(out_channels, out_channels)
        # Batch normalization layer after the second convolution
        self.bn2 = nn.BatchNorm2d(out_channels)
        # Optional downsample function (not used in this implementation)
        self.downsample = downsample
        # Define the skip connection (residual connection)
        # By default, it's an identity function that passes the input unchanged
        self.skip = nn.Identity()  # function returns a module that simply passes its input through unchanged.
        # If the input and output channels don't match, or the stride is not 1,
        # create a sequential module with a 1x1 convolution and batch normalization
        if in_channels != out_channels or stride != 1:
            # Create a sequential module with a 1x1 convolution and batch normalization
            self.skip = nn.Sequential(
                # 1x1 convolution to match the number of channels and adjust the spatial dimensions if stride is not 1
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                # Batch normalization layer to normalize the output of the 1x1 convolution
                nn.BatchNorm2d(out_channels)
            )
        # Store the input and output channel numbers for later use
        self.in_channels = in_channels
        self.out_channels = out_channels

    # Define the forward pass of the ResidualBlock
    def forward(self, x):
        # Save the input as the identity for the residual connection
        identity = x
        # Pass the input through the first convolution
        out = self.conv1(x)
        # Normalize the output of the first convolution
        out = self.bn1(out)
        # Apply the ReLU activation function
        out = self.relu(out)
        # Pass the output through the second convolution
        out = self.conv2(out)
        # Normalize the output of the second convolution
        out = self.bn2(out)
        # Apply the skip connection to the identity (input)
        identity = self.skip(identity)  # Apply the skip connection
        # Add the identity (residual) to the output
        out += identity
        # Apply the ReLU activation function to the combined output
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, num_blocks):
        super(ResNet, self).__init__()
        # Set the initial number of channels
        self.in_channels = 32
        # Create the initial 3x3 convolution layer with 3 input channels for RGB images
        self.conv = conv3x3(3, self.in_channels)  # Change the input channels to 3
        # Add a batch normalization layer after the initial convolution
        self.bn = nn.BatchNorm2d(32)
        # Define the ReLU activation function used in the network
        self.relu = nn.ReLU(inplace=True)
        # Create a list to hold the residual blocks
        layers = []
        # Iterate over the number of blocks specified for each stage
        for num in num_blocks:
            # Set the stride to 2 for downsampling, except for the first stage where stride is 1
            stride = 2 if self.in_channels != 32 else 1
            # Add the first ResidualBlock with the specified stride for each stage
            layers.append(ResidualBlock(self.in_channels, self.in_channels * 2, stride=stride))
            # Update the number of input channels for the next stage
            self.in_channels *= 2
            # Add the remaining ResidualBlocks for each stage with stride=1
            for _ in range(num - 1):
                layers.append(ResidualBlock(self.in_channels, self.in_channels))
        # Create a sequential container for the residual blocks
        self.res_blocks = nn.Sequential(*layers)
        # Add an adaptive average pooling layer to reduce the spatial dimensions to 1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Define the fully connected layer with 10 output units for the CIFAR-10 dataset
        self.fc = nn.Linear(self.in_channels, 10)  # adjust output size based on number of blocks
    # Define the forward pass of the ResNet
    def forward(self, x):
        # Pass the input through the initial convolution
        out = self.conv(x)
        # Normalize the output of the initial convolution
        out = self.bn(out)
        # Apply the ReLU activation function
        out = self.relu(out)
        # Pass the output through the residual blocks
        out = self.res_blocks(out)
        # Apply the adaptive average pooling
        out = self.avgpool(out)
        # Flatten the output tensor
        out = out.view(out.size(0), -1)
        # Pass the flattened output through the fully connected layer
        out = self.fc(out)
        return out
    
######################################################################################
##############################Model training-testing##################################
######################################################################################

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 25
batch_size = 100
learning_rate = 0.001

# create an instance of the ResNet class
num_blocks = [2, 2, 2]  # 3 groups, each with 2 residual blocks
resnet = ResNet(num_blocks)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet.parameters(), lr=learning_rate)

# Initialize the learning rate decay factor
decay = 0
# Set the model to training mode (enables dropout and batch normalization layers)
resnet.train()
# Iterate through the specified number of training epochs
for epoch in range(num_epochs):
    # Decay the learning rate every 20 epochs
    if (epoch+1) % 20 == 0:
        decay += 1
        # Update the learning rate in the optimizer's parameter group
        optimizer.param_groups[0]['lr'] = learning_rate * (0.5**decay)
        # Print the new learning rate
        print("The new learning rate is {}".format(optimizer.param_groups[0]['lr']))
    # Iterate through the batches in the train_loader
    for i, (images, labels) in enumerate(tqdm(train_loader)):
        # Move the images and labels to the appropriate device (CPU or GPU)
        images = images  # .to(device)
        labels = labels  # .to(device)
        # Perform a forward pass through the model to get the predicted class probabilities
        outputs = resnet(images)
        # Calculate the loss between the predicted probabilities and the ground truth labels
        loss = criterion(outputs, labels)
        # Zero the gradients before backpropagation
        optimizer.zero_grad()
        # Perform backpropagation to calculate the gradients
        loss.backward()
        # Update the model parameters using the calculated gradients
        optimizer.step()
        # Print the training loss every 100 steps
        if (i+1) % 100 == 0:
            print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                  .format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
           

##Test the model##
# Set the model to evaluation mode (disables dropout and batch normalization layers)
resnet.eval()
# Disable gradient calculations for faster evaluation and reduced memory usage
with torch.no_grad():
    # Initialize variables to track the number of correct predictions and total samples
    correct = 0
    total = 0
    # Iterate through the batches in the test loader
    for images, labels in test_loader:
        # Move the images and labels to the appropriate device (CPU or GPU)
#        images = images.to(device)
#        labels = labels.to(device)
        # Perform a forward pass through the model to get the predicted class probabilities
        outputs = resnet(images)
        # Find the class with the highest probability for each sample
        _, predicted = torch.max(outputs.data, 1)
        # Update the total number of samples
        total += labels.size(0)
        # Update the number of correct predictions
        correct += (predicted == labels).sum().item()

    # Calculate and print the accuracy of the model on the test images
    print('Accuracy of the model on the test images: {} %'.format(100*correct/total))

#################################
####################
############
#####
###
#

# Test the model
resnet.eval()

# Choose 10 random indices from the test dataset
random_indices = random.sample(range(len(test_dataset)), 20)
# Subset the test dataset using the 10 random indices
test_subset = torch.utils.data.Subset(test_dataset, random_indices)
test_subset_loader = torch.utils.data.DataLoader(test_subset, batch_size=1)

correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_subset_loader:
        outputs = resnet(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Print the true label, predicted label, and whether the prediction is correct
        print(f'True Label: {labels.item()}, Predicted Label: {predicted.item()}, Correct: {predicted.item() == labels.item()}')

# Calculate and print the accuracy of the model on the 10 test samples
print('Accuracy of the model on the 10 random test samples: {} %'.format(100 * correct / total))

#######################SAVE MODEL#########################
# Save Model Weights
torch.save(resnet.state_dict(), 'ResNet_model_weights.pth')