import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Define the path where the dataset will be stored
data_path = r'C:\Users\behna\OneDrive\Documents\GitHub\Pytorch\4.convolutions'

# Download and load the MNIST dataset
train_dataset = datasets.MNIST(root=data_path, train=True, download=True, transform=None)
test_dataset = datasets.MNIST(root=data_path, train=False, download=True, transform=None)

# you can use the transforms module to resize, crop, flip, rotate, 
# or normalize images, and also to convert images to tensors 
# or apply other custom transformations.