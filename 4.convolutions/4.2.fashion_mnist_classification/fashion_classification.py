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
data_path = r'C:\Users\behna\OneDrive\Documents\GitHub\Pytorch\4.convolutions\4.2.fashion_mnist_classification\fashion_dataset'

# pre-computed mean and std of mnist
mean = 0.2860
std = 0.3530

# transformation of data
# Transformations can include data augmentation techniques such as
#  random cropping or flipping, or normalization techniques such as 
# scaling or mean subtraction. If not specified, the default transform is to convert 
# the images to tensors and normalize them to the range [0, 1]

# NORMALIZATION PROCESS
# input[channel] = (input[channel] - mean(channel)) / standard deviation

transform = transforms.Compose([
#    transforms.RandomCrop(size=14),
#    transforms.RandomRotation(degrees=90), # randomly rotate the images by up to 20 degrees
#    transforms.RandomHorizontalFlip(p=0.3),# randomly flip the images horizontally with a probability of 0.3
    transforms.ToTensor(),                 # convert images to tensors
    transforms.Normalize((mean,), (std,))  # normalize the tensor with mean and standard deviation
                                ])

# Download and load the MNIST datase
# datasets.'xx' can search for all types of image datasets
train_dataset = datasets.FashionMNIST(
                                        root=data_path, 
                                        train=True, 
                                        download=True, 
                                        transform=transform
                                    )

# # Compute the mean and standard deviation of the dataset
# train_loader = torch.utils.data.DataLoader(train_dataset, 
#                                            batch_size=len(train_dataset)
#                                            )
# data = next(iter(train_loader))
# mean, std = data[0].mean(), data[0].std()

# print("Mean:", mean)
# print("Std:", std)

test_dataset = datasets.FashionMNIST(
                                        root=data_path, 
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
image, label = train_dataset[1]

# must convert 2D tensor (1, 28, 28) colour_channel, dim, dim
# to 28x28 image removing colour
image = image.numpy()[0]
# Display the image and label
plt.imshow(image, cmap='gray')
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
# check layer shapes and weight
model.conv1[0].weight.shape

# Print model's state_dict
# provides weights and bias at each stage
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# setup optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
##########################################################
##########################################################
##########################################################
#Understand what's happening before training
iteration = 0
correct = 0

for i, (inputs,labels) in enumerate(train_loader):
        
    print("For one iteration, this is what happens:")
    print("Input Shape:",inputs.shape)
    print("Labels Shape:",labels.shape)
    output = model(inputs)
    print("Outputs Shape",output.shape)
    _, predicted = torch.max(output, 1)
    print("Predicted Shape",predicted.shape)
    print("Predicted Tensor:")
    print(predicted) # provide prediction/batch -> 64 predictions
    correct += (predicted == labels).sum()
    break

print(correct)

##########################################################
##################### model training######################
##########################################################
# initialize lists to store loss and accuracy
train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []

num_epochs = 10
patience = 4  # number of epochs to wait before early stopping
best_test_loss = np.inf
best_model_params = None
early_stopping_counter = 0

# train the model
for epoch in range(num_epochs):
    # training
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        # move data and target to device
        data, target = data.to(device), target.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(data)
        loss = nn.CrossEntropyLoss()(outputs, target)
        loss.backward()
        optimizer.step()

        # calculate training accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        # add current batch loss to running loss
        running_loss += loss.item()

    # calculate average training loss and accuracy for current epoch
    avg_train_loss = running_loss / len(train_loader)
    avg_train_accuracy = 100 * correct / total
    train_loss.append(avg_train_loss)
    train_accuracy.append(avg_train_accuracy)

    # testing
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # move data and target to device
            data, target = data.to(device), target.to(device)

            # forward pass
            outputs = model(data)
            loss = nn.CrossEntropyLoss()(outputs, target)

            # calculate test accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            # add current batch loss to running loss
            running_loss += loss.item()

        # calculate average test loss and accuracy for current epoch
        avg_test_loss = running_loss / len(test_loader)
        avg_test_accuracy = 100 * correct / total
        test_loss.append(avg_test_loss)
        test_accuracy.append(avg_test_accuracy)

        # check for early stopping
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_model_params = model.state_dict()
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f"Stopping early after {epoch+1} epochs.")
                break

    # print training and testing loss and accuracy
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.2f}%, Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_accuracy:.2f}%")

    # check for early stopping before starting the next epoch
    if early_stopping_counter >= patience:
        break

##########################################################
##########################################################
##########################################################
# visualizing plot dimensions
def plot_metrics(train_loss, test_loss, train_accuracy, test_accuracy):
    fig, axs = plt.subplots(1, 2, figsize=(12, 8))
    
    axs[0].plot(train_loss, '-b', label='Training loss')
    axs[0].plot(test_loss, '-r', label='Testing loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend(loc='upper right')
    axs[0].set_title('Training and Testing Loss')
    
    axs[1].plot(train_accuracy, '-b', label='Training accuracy')
    axs[1].plot(test_accuracy, '-r', label='Testing accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend(loc='lower right')
    axs[1].set_title('Training and Testing Accuracy')
    
    plt.tight_layout()
    plt.show()


plot_metrics(train_loss, test_loss, train_accuracy, test_accuracy)
##########################################################
##########################################################
##########################################################
#Understand what's happening after training
iteration = 0
correct = 0

for i,(inputs,labels) in enumerate (train_loader):
        
    print("For one iteration, this is what happens:")
    print("Input Shape:",inputs.shape)
    print("Labels Shape:",labels.shape)
    output = model(inputs)
    print("Outputs Shape",output.shape)
    _, predicted = torch.max(output, 1)
    print("Predicted Shape",predicted.shape)
    print("Predicted Tensor:")
    print(predicted) # provide prediction/batch -> 64 predictions
    correct += (predicted == labels).sum()
    break
print(correct)
# model can get 61 out of 64 images correct

a = torch.tensor([1, 2, 3])
b = torch.tensor([2, 2, 4])
a==b
(a==b).sum().item()

##########################################################
#####################TEST MODEL###########################
##########################################################
# resize a tensor to 4D tensor to pass via the model 
# Choose a random set of examples from the test dataset
indices = random.sample(range(len(test_dataset)), 10)

for i in indices:
    # Get the image and label
    img = test_dataset[i][0].resize_((1, 1, 28, 28))
    label = test_dataset[i][1]

    # Make the prediction
    output = model(img)
    _, predicted = torch.max(output, 1)

    # Print the results
    print("Prediction for example {} is: {}".format(i, predicted.item()))
    print("Actual label is: {}".format(label))

##########################################################
#######################SAVE MODEL#########################
##########################################################
# save the trained model weights and bias for transfer using:
#torch.save(model.state_dict(), 'cnn_model_weights.pth')

# save the model architecture, including its weights and bias
#torch.save(model, 'cnn_model.pth')