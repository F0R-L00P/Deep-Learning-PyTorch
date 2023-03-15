import torch
import torch.nn as nn
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
image, label = train_dataset[2]

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
test_loader = torch._utils.data.DataLoader(test_dataset,
                                           batch_size=batch_size,
                                           shuffle=False
                                           )

