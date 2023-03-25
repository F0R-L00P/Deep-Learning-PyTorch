import os
import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm

from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

##########################################################
# use a powerful pretrained model i.e. AlexNet/ResNets?etc 
# 1) freez all layers
# 2) remove the last layer 
# 3) train and reconstruct last layer using your own data
#       -> image net has 1000 classes, your data has 5
#       -> change last layer to reflect 5
##########################################################
# to normalize input:
# loop over all images and obtain mean/std using pytorch
# --> normalize per channel
# --> (input[channel] - mean[channel]) / std[channel]
train_transform = transforms.Compose([
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                             std=[0.229, 0.224, 0.225])
                                    ])

val_transform = transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                             std=[0.229, 0.224, 0.225])
                                    ])

# define path
data_path = "C:/Users/behna/OneDrive/Desktop/PYTHON VENV CODE/hymenoptera_data/hymenoptera_data"

# Load the images using ImageFolder class, and apply the transforms
train_dataset = datasets.ImageFolder(os.path.join(data_path, "train"), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(data_path, "val"), transform=val_transform)

# Print the number of images in each dataset
print("Number of training images:", len(train_dataset))
print("Number of validation images:", len(val_dataset))

# load both training set and validation set in the dataloader class
batch = 32
workers = 10

train_loader = DataLoader(train_dataset,
                           batch_size=batch, 
                           shuffle=True, 
                           num_workers=workers
                        )

val_loader = DataLoader(val_dataset, 
                        batch_size=batch, 
                        shuffle=False, 
                        num_workers=workers
                        )

######Get a batch of training data####
#dataiter = iter(train_loader)
#images, labels = next(dataiter)

######Expect batch of 32, 3 channels image h=224 by w=224######
# Print the shape of the image and label tensors
#print("Image tensor shape:", images.shape)
#print("Label tensor shape:", labels.shape)


#################################################################
#############################BUILD MODEL#########################
#################################################################
resnet18 = torchvision.models.resnet18(weights=torchvision.models.resnet.ResNet18_Weights.IMAGENET1K_V1)

# check model parametr and srchitecture
print(resnet18.parameters)

# Freeze all layers in the network
for param in resnet18.parameters():
    param.requires_grad = False

# check input features to the last layer (linear)
# input & output
print(resnet18.fc)

# the output feature must be reconstructed to classify 2 classes only
# output neurons should be changed from 1000 to 2
# Changing the output features of the last fully connected layer to 2
resnet18.fc = nn.Linear(in_features=512, out_features=2, bias=True)

# check model architecture reflecting the change
print(resnet18.fc)

#################################################################
#################################################################
#################################################################
if torch.cuda.is_available():
    resnet18 = resnet18.cuda()

# FOR 1 ITERATION
# model output before training
iteration = 0
correct = 0
for inputs, labels in tqdm(train_loader, desc="Processing"):
    if iteration == 1:
        break
    if torch.cuda.is_available():
        inputs = inputs.cuda()
        labels = labels.cuda()
    output = resnet18(inputs)
    _, predicted = torch.max(output, 1)
    correct += (predicted == labels).sum()
    print(f"{correct.item()} Correct Predictions out of {len(predicted)}")

    iteration += 1
#################################################################
#############################Training MODEL######################
#################################################################
# Set requires_grad to True for the fc layer
for param in resnet18.fc.parameters():
    param.requires_grad = True

# Set up the criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(resnet18.fc.parameters(), lr=0.001, momentum=0.9)
# Set up the learning rate scheduler
# Reduce the learning rate by a factor of 0.1 every 30 epochs
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)  

# Training loop (example for one epoch)
num_epochs = 20
for epoch in range(num_epochs):
    resnet18.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for inputs, labels in pbar:
        inputs = inputs.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()

        outputs = resnet18(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Update progress bar
        pbar.set_postfix(loss=loss.item(), acc=(correct / total) * 100)

    # Update the learning rate at the end of the epoch
    scheduler.step()

    # Calculate average loss and accuracy for the epoch
    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100

    # Print training loss and accuracy after each epoch
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

# Test the model
resnet18.eval()
with torch.no_grad():
    correct = 0
    total = 0
    running_loss = 0.0
    progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation")

    for i, (images, labels) in progress_bar:
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        outputs = resnet18(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Calculate and print the average loss and accuracy after each iteration
        avg_loss = running_loss / (i + 1)
        avg_accuracy = 100 * correct / total
        progress_bar.set_postfix({"Loss": avg_loss, "Accuracy": avg_accuracy})

print('Test Accuracy: {:.3f} %'.format(100 * correct / total))