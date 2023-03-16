import torch
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

# Define the path where the dataset will be stored
data_path = r'C:\Users\behna\OneDrive\Documents\GitHub\Pytorch\3.2.conv_pytorch_dataset'

# pre-computed mean and std of mnist
mean = 0.1307
std = 0.3081

# transformation of data
# Transformations can include data augmentation techniques such as
#  random cropping or flipping, or normalization techniques such as 
# scaling or mean subtraction. If not specified, the default transform is to convert 
# the images to tensors and normalize them to the range [0, 1]

# NORMALIZATION PROCESS
# input[channel] = (input[channel] - mean(channel)) / standard deviation


transform = transforms.Compose([
    transforms.RandomRotation(degrees=20), # randomly rotate the images by up to 20 degrees
    transforms.RandomHorizontalFlip(p=0.3),# randomly flip the images horizontally with a probability of 0.3
    transforms.ToTensor(),                 # convert images to tensors
    transforms.Normalize((mean,), (std,))  # normalize the tensor with mean and standard deviation
                                ])

# Download and load the MNIST datase
# datasets.'xx' can search for all types of image datasets
train_dataset = datasets.MNIST(root=data_path, 
                               train=True, 
                               download=True, 
                               transform=transform
                               )
test_dataset = datasets.MNIST(root=data_path, 
                              train=False, 
                              download=True, 
                              transform=transform
                              )

# check data shape
# 28x28 image for 60k train
train_dataset.data.shape
# 28x28 image for 10k test
test_dataset.data.shape

# visualize data 
# Select a random image from the TRAIN - dataset
image, label = train_dataset[0]

# must convert 2D tensor (1, 28, 28) colour_channel, dim, dim
# to 28x28 image removing colour
image = image.numpy()[0]
# Display the image and label
plt.imshow(image, cmap='gray')
plt.title(f"Label: {label}")
plt.axis("off")
plt.show()

# check corresponding label 
print(label)

# setup dataloader for training process
# Create the DataLoader with batch size 64, shuffling
batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True
                                           )
test_loader = torch.utils.data.DataLoader(test_dataset,
                                           batch_size=batch_size,
                                           shuffle=False
                                           )

# lets visualize the loader
# as batch is 64, instances are expected to be 60000\64 & 10000\64
print(f'Train Loader: {len(train_loader)} | Test Loader: {len(test_loader)}')

#########################################################
##################### DEFINE MODEL ######################
class CNN(nn.Module):
    def __init__(self):
        # build constructor-parent class nn.Module
        super(CNN, self).__init__()  
        # conv layer with batch norm and ReLU activation
        self.conv1 = nn.Sequential(  
            # Convolutional layer with 1 input channel, 8 output channels, and 3x3 kernel size
            # if image was RBG input would be 3
            # pad image to keep original size calculated as
                #(kernal_size -1)\2
            #[(input_size - filter_size + 2(padding) / stride) +1] --> 
            # [(28-3+2(1)/1)+1] = 28 (padding type is same)
            # input size = output size
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1), 
            # Batch normalization layer for 8 channels
            nn.BatchNorm2d(8),
            # ReLU activation layer
            nn.ReLU(),  
        )

        # max pooling layer with 2x2 kernel size
        # output sizeof image should decrease after maxpooling
        # (input_size - kernel_size)/stride) + 1 -->
        # (28 - 2) / 2) + 1 = 14
        self.pool1 = nn.MaxPool2d(kernel_size=2)  
        # conv layer with batch norm and ReLU activation
        self.conv2 = nn.Sequential(  
            # Convolutional layer with 8 input channels, 32 output channels, and 5x5 kernel
            # image output size output_size = ((28 - 5 + 24) / 1) + 1
            # 28
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=5, padding=2),  
            # Batch normalization layer for 32 channels
            nn.BatchNorm2d(32),
            # ReLU activation layer
            nn.ReLU(),  
        )

        # Define the second max pooling layer with 2x2 kernel size
        # second maxpool will reduce image size to 7x7
        # (14 - 2)/2 + 1 = 7
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        # fully connected layer input size of 32*5*5 and output size of 600
        self.fc1 = nn.Linear(in_features=(32 * 7 * 7), 
                             out_features=600)
        self.dropout = nn.Dropout(p=0.5)
        # fully connected layer  600 * 5 * 5 input - 10 output
        self.fc2 = nn.Linear(in_features=600, 
                             out_features=10)  

    def forward(self, x):
        x = self.conv1(x)  # conv layer w batch norm and ReLU activation
        x = self.pool1(x)  # max pooling layer
        x = self.conv2(x)  # conv layer w batch norm and ReLU activation
        x = self.pool2(x)  # max pooling layer
        # Flatten the output of the second max pooling layer to a 1D tensor
        # can use explicit value using btach_size vs -1
        x = x.view(-1, 32 * 7 * 7)  
        x = self.fc1(x)  # fully connected layer
        x = self.dropout(x)
        x = self.fc2(x)  # fully connected layer
        return x


# setup cuda process
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create the model and optimizer
model = CNN()

# Move model to device
model.to(device)

# check model weight matrix from each layer
x = torch.randn(1, 1, 28, 28)
model(x)

model
# check layer shapes
model.conv1[0]

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    # Set the model to training mode
    model.train()
    # Iterate over the batches of training data
    for images, labels in train_loader:
        # Zero out the gradients
        optimizer.zero_grad()
        # Forward pass
        output = model(images)
        # Compute the loss
        loss = nn.CrossEntropyLoss()(output, labels)
        # Backward pass
        loss.backward()
        # Update the weights
        optimizer.step()

    # Set the model to evaluation mode
    model.eval()
    # Compute the accuracy on the test set
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in test_loader:
            output = model(images)
            _, predicted = torch.max(output, dim=1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    accuracy = total_correct / total_samples

    # Print the epoch number and test accuracy
    print(f"Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {accuracy:.4f}")