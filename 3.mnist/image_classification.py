# objective
#1) flatten image
#2) build network architecture

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from torch.autograd import Variable


input_size = 784        #Number of input neurons (image pixels)
hidden_size = 400       #Number of hidden neurons
out_size = 10           #Number of classes (0-9) 
epochs = 10            #How many times we pass our entire dataset into our network 
batch_size = 100        #Input size of the data during one iteration 
learning_rate = 0.001   #How fast we are learning

# train test split
train_data = datasets.MNIST(
                root="data",
                train=True,
                download=True,
                transform=transforms.ToTensor()
                )

test_data = datasets.MNIST(
                root="data",
                train=False,
                transform=transforms.ToTensor()
                )

# setup dataloaders
train_loader = DataLoader(
                    dataset=train_data,
                    batch_size=batch_size,
                    shuffle=True
                    )

test_loader = DataLoader(
                    dataset=test_data,
                    batch_size=batch_size,
                    shuffle=False
                    ) 

# building network model
# when initializing the class the architecture gets initalized first
class NeuralNet(nn.Module):
    def __init__(self, input_layer, hidden_layer, output_layer):
        super(NeuralNet, self).__init__()
        # flatten images
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, out_size)
        self.activation = nn.ReLU()

    def forward(self, X):
        output_var1 = self.activation(self.input_layer(X))
        output_var2 = self.activation(self.hidden_layer(output_var1))
        output_var3 = self.output_layer(output_var2)

        return output_var3

# Create an object of the class, which represents our network 
net = NeuralNet(input_size, hidden_size, out_size)
# view network architecture and layer
print(net.get_parameter)
# acossiated layer weight
print(list(net.parameters()))

CUDA = torch.cuda.is_available()
# if GPU is available then switch tensors on GPU
if CUDA:
    net = net.cuda()
# The loss function. The Cross Entropy loss comes along with Softmax. 
# Therefore, no need to specify Softmax as well
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

##############################
# VISUALIZING the train_loader
#############################
for i, (images, label) in enumerate(train_loader):
    print(images.size())
    images = images.view(-1, 784) # this will flatten out the image
    print(images.size())


###################
# TRAIN the network
###################
# training process must be defined step by step

# first loop is epoch
for epoch in range(epochs):
    correct_train = 0
    running_loss = 0
    # second loop is iteration
    # each iteration takes batch of 100 and applies the following
    for i, (images, labels) in enumerate(train_loader):   
        # Flatten the image from size (batch,1,28,28) --> (100,1,28,28) 
        # where 1 represents the number of channels (grayscale-->1),
        # to size (100,784) and wrap it in a variable
        images = images.view(-1, 28*28)    
        if CUDA:
            images = images.cuda()
            labels = labels.cuda()
        # clear gradient
        optimizer.zero_grad() 
        outputs = net(images)                             # complete forward pass      
        _, predicted = torch.max(outputs.data, 1)         # max provides index of highest value per row                              
        correct_train += (predicted == labels).sum()      # compare 2 tensos, sums number of the index equal values
        loss = criterion(outputs, labels)                 # Difference between the actual and predicted (loss function)
        running_loss += loss.item()                       # loss.item() fetchs tensor value
        loss.backward()                                   # Backpropagation
        optimizer.step()                                  # Update the weights
        
    print('Epoch [{}/{}], Training Loss: {:.3f}, Training Accuracy: {:.3f}%'.format
          (epoch+1, epochs, running_loss/len(train_loader), (100*correct_train.double()/len(train_data))))
print("DONE TRAINING!")

# suming EXAMPLE
t1 = torch.tensor([1, 2, 3, 4])
t2 = torch.tensor([1, 8, 8, 4])
print(t1 == t2)
print((t1 == t2).sum())

###################
# TEST the network
###################
# no_grad():
    # It means any tensor with gradient currently attached with the current 
    # computational graph is now detached from the current graph (DO NOT updte, weights)
with torch.no_grad():
    correct = 0
    for images, labels in test_loader:
        if CUDA:
            images = images.cuda()
            labels = labels.cuda()
        images = images.view(-1, 28*28)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / len(test_data)))