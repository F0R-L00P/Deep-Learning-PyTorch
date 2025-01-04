import os
import torch
import torchvision
import torch.nn as nn

from torch.autograd import Variable
from torchvision.models.vgg import VGG16_Weights
from torchvision import datasets, models, transforms


import json
import scipy.misc
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

###############################################################
###############################################################
# get the current working directory
cwd = os.getcwd()

# define transoms object to augment image
transform = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                    std=[0.229, 0.224, 0.225])
                            ])

# get the current working directory
cwd = os.getcwd()
# define the file name
file_name = "dog.jpg"

# create the full file path
file_path = os.path.join(cwd, file_name)

# open the image
image = Image.open(file_path)
plt.imshow(image);

###############################################################
###############################################################
# load pretrained model to obtain feature maps after convolutions
vgg_model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

# run model on GPU
vgg_model = vgg_model.cuda()

# review model parameters and architecure
print(vgg_model.parameters)
# lets view final fully connected layer of the classifier
# last layer should have input features and output for 1000 classes
print(vgg_model.classifier[6]) # or use [-1]

# lets apply the transformation and view the image after transformation
image = transform(image)
# 3 channels, 224x224 pixels
image.shape

# given the shape of the tensor, [3, 224, 224]
# to pass through a network, a 4th dimension must be added representing the batch
# this can be completed with
image = image.unsqueeze(0)
# givin [1, 3, 224, 224]
image.shape
###############################################################
###############################################################
# lets run image through the vgg network
model_output = vgg_model(image.cuda())

# view model output, expected output shoul be 1 image that is flattened
# with dimensions [1, 1000]
print(model_output.shape)

# the batch can be dropped again using the squeeze parameter
model_output = model_output.squeeze(0)
print(model_output.shape)
###############################################################
###############################################################
# lets load the class label index
# using open will close file after reading
with open('imagenet_class_index.json', 'r') as f:
    labels = json.load(f)
# view sample labels, using random key
print(labels['20'])

# model prediction is obtained by [assing image through the network, 
# softmax layer produces 1000 probability outputs, we need to get the max value
# coressponding the to prediction
index = model_output.max(0)
# out put is 2 tensors
# first tensor is the score
# second is the index
print(index)

# lets access the index value of the second tensor
# note convert output to string to search the index of the json file for the image
index = str(index[1].item())

# search for the index in the label
labels[index] # BOOM ;-)

###############################################################
###############################################################
# the goal is to visualize the model features, therefore,
# lets get the features of the VGG network
print(vgg_model.features)
# we are intereste din the models, therefore cast to list and access the models
model_list = list(vgg_model.features.modules())
model_list

# OBJECTIVE
# This loop essentially obtains an image 
# representation from each layer in the model
outputs = []
names = []
for layer in model_list[1:]:
    image = layer(image.cuda())
    outputs.append(image.cuda())
    names.append(str(layer))
# check first layer output
print(names[0]) 
print(outputs[0].shape)

# lets check all featuremaps as passing through each layer
for feature_map in outputs:
    print(feature_map.shape)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define process converting 3d tensor to 2d
processed = []
for feature_map in outputs:
    feature_map = feature_map.squeeze(0)
    gray_scale = torch.sum(feature_map, 0, keepdim=True)
    gray_scale.div_(feature_map.shape[0])
    processed.append(gray_scale.squeeze(0).data.cpu().numpy())

# visualize the feature maps
# will be used to display the feature maps
fig = plt.figure(figsize=(30, 60))
for i in range(len(processed)):
    #adds a new subplot to the figure, with a grid of 8 rows and 4 columns
    #specifies the position of the subplot within the grid
    view = fig.add_subplot(8, 4, i+1)
    #displays the processed feature map at index i in the subplot
    image_plot = plt.imshow(processed[i])
    #turn off the axis labels and ticks
    plt.axis('off')

    #extracts the name of the layer that produced the feature map
    layer_name = names[i].split('(')[0]
    #checks if the layer name contains parentheses 
    # (indicating a layer that produced a feature map), 
    # and is not a ReLU or pooling layer
    if '(' in names[i] and 'ReLU' not in layer_name and 'Pool' not in layer_name:
        #extracts the input and output channel dimensions of the layer
        layer_channel = names[i].split('(')[1].split(',')
        #checks if both the input and output channel 
        # dimensions are available for the layer
        if len(layer_channel) >= 2:
            #sets the title if both values available
            view.set_title(layer_name + f'\ninput: {layer_channel[0]}, output: {layer_channel[1]}', fontsize=30)
        else:
            view.set_title(layer_name, fontsize=30)
    else:
        view.set_title(layer_name, fontsize=30)

# Save the figure
plt.savefig("feature_maps.png", dpi=300, bbox_inches='tight')
plt.show()