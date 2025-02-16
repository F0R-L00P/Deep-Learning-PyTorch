import torch
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid

# Define the path where the dataset will be stored
data_path = r'4.convolutions\4.2.fashion_mnist_classification\fashion_dataset'

# pre-computed mean and std of mnist
mean = 0.2860
std = 0.3530

# transformation of data
transform = transforms.Compose([
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
batch_size = 128
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
#########################################################
############CNN miniVGGNet architecture #################
#########################################################
#VGGNet architecture proposed in the paper 
# "Very Deep Convolutional Networks for Large-Scale Image Recognition" by 
# Simonyan and Zisserman does not include batch normalization. 
# However, I will modify the VGGNet architecture to include 
# batch normalization layers to improve its performance.
class MiniVGGNet(nn.Module):
    def __init__(self, dropout, numn_classes = 10):
        # build constructor-parent class nn.Module
        super(MiniVGGNet, self).__init__()  
        # conv layer with batch norm and ReLU activation
        self.conv_layer = nn.Sequential(  
            # Convolutional layer with 1 input channel, 32 output channels, and 3x3 kernel size
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            # Batch normalization layer for 32 channels
            nn.BatchNorm2d(32),
            # ReLU activation layer
            nn.ReLU(inplace=True),
            # Convolutional layer with 1 input channel, 32 output channels, and 3x3 kernel size
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1), 
            # Batch normalization layer for 32 channels
            nn.BatchNorm2d(32),
            # ReLU activation layer
            nn.ReLU(inplace=True),
            # max pooling layer with 2x2 kernel size
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Convolutional layer with 1 input channel, 32 output channels, and 3x3 kernel size
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), 
            # Batch normalization layer for 64 channels
            nn.BatchNorm2d(64),
            # ReLU activation layer
            nn.ReLU(inplace=True),
            # Convolutional layer with 1 input channel, 32 output channels, and 3x3 kernel size
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), 
            # Batch normalization layer for 64 channels
            nn.BatchNorm2d(64),
            # ReLU activation layer
            nn.ReLU(inplace=True),
            # max pooling layer with 2x2 kernel size
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
        # fully connected layer input size of 32*5*5 and output size of 600
        self.classifier = nn.Sequential(
            nn.Linear(in_features=(64 * 7 * 7), out_features=512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),    
            # fully connected layer  512 * 7 * 7 input - 10 output
            nn.Linear(in_features=512, out_features=numn_classes) 
        )

    def forward(self, x):
        x = self.conv_layer(x)  # conv layer w batch norm and ReLU activation
        x = x.view(x.size(0), -1)
        x = self.classifier(x)  # fully connected layer
        return x
##########################################################
##################### model training######################
##########################################################
# setup cuda process
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_epochs = 40
patience = 5  # number of epochs to wait before early stopping
def train_evaluate(model, 
                   train_loader, 
                   test_loader, 
                   criterion, 
                   optimizer, 
                   num_epochs=num_epochs, 
                   patience=patience):
    
    # move model and optimizer to device
    model.to(device)

    # initialize lists to store loss and accuracy
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []

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
            data = data.clone().detach().requires_grad_(True).to(device) 
            target = target.clone().detach().to(device)

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
            test_accs = []
            for batch_idx, (data, target) in enumerate(test_loader):
                # move data and target to device
                data = data.clone().detach().requires_grad_(True).to(device) 
                target = target.clone().detach().to(device)

                # forward pass
                outputs = model(data)
                loss = nn.CrossEntropyLoss()(outputs, target)

                # calculate test accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

                # add current batch loss to running loss
                running_loss += loss.item()

                # calculate test accuracy for current batch
                batch_acc = 100 * (predicted == target).sum().item() / target.size(0)
                test_accs.append(batch_acc)

            # calculate average test loss and accuracy for current epoch
            avg_test_loss = running_loss / len(test_loader)
            avg_test_accuracy = sum(test_accs) / len(test_accs)
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
        print(f"Epoch [{epoch+1}/{num_epochs}], \
            Train Loss: {avg_train_loss:.4f}, \
            Train Accuracy: {avg_train_accuracy:.2f}%, \
            Test Loss: {avg_test_loss:.4f}, \
            Test Accuracy: {avg_test_accuracy:.2f}%"
            )

    return train_loss, test_loss, train_accuracy, test_accuracy, avg_test_accuracy
##########################################################
#####################HYPERPARAMETERS######################
##########################################################
# Perform grid search
best_accuracy = 0
best_params = None

# Define hyperparameter search space
param_grid = {
    'dropout': [0.5],
    'learning_rate': [1e-3],
    'optimizer': ['Adam']
}


for params in ParameterGrid(param_grid):
    print("Current parameters:", params)
    # Create model with the current parameters
    model = MiniVGGNet(dropout=params['dropout'])
    
    model.to(device)
    # Set up the optimizer
    learning_rate = params['learning_rate']
    if params['optimizer'] == 'Adam':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    if params['optimizer'] == 'AdamW':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Set up the loss function
    criterion = nn.CrossEntropyLoss()   

    # Train and evaluate the model
    # Train and evaluate your model
    train_loss, test_loss, train_accuracy, test_accuracy, avg_test_accuracy = train_evaluate(
        model, train_loader, test_loader, criterion, optimizer, num_epochs=num_epochs, patience=patience
        )

    # Update the best model if the accuracy is improved
    if avg_test_accuracy > best_accuracy:
        best_accuracy = avg_test_accuracy
        best_params = params

print("Best accuracy:", best_accuracy) # 93.4
print("Best parameters:", best_params) # {'dropout': 0.5, 'learning_rate': 0.001, 'optimizer': 'Adam'}
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
# Understand what's happening after training
# manual check for model accuracy, compare results to first output
iteration = 0
correct = 0

for i,(inputs,labels) in enumerate (train_loader):
        
    print("For one iteration, this is what happens:")
    print("Input Shape:",inputs.shape)
    print("Labels Shape:",labels.shape)
    output = model(inputs.cuda())
    print("Outputs Shape",output.shape)
    _, predicted = torch.max(output.cuda(), 1)
    print("Predicted Shape",predicted.shape)
    print("Predicted Tensor:")
    print(predicted) # provide prediction/batch -> 64 predictions
    correct += (predicted == labels.cuda()).sum()
    break

print(correct)

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
    output = model(img.cuda())
    _, predicted = torch.max(output.cuda(), 1)

    # Print the results
    print(f"Prediction for example {i} is: {predicted.item()}")
    print(f"Actual label is: {label}")

##########################################################
#######################SAVE MODEL#########################
##########################################################
# save the trained model weights and bias for transfer using:
#torch.save(model.state_dict(), 'cnn_model_weights.pth')

# save the model architecture, including its weights and bias
#torch.save(model, 'cnn_model.pth')