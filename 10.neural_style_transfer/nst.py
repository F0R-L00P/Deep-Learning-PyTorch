# to complet a neural style transfer we need:
# 1) need a pre-trained NN (i.e. VGG, ResNet, etc)
# 2) content image to transfer style to
# 3) style image transferring the style from
# 4) output -> generated image with final result
import io
import os
import torch
import torchvision
import torch.nn as nn

from torchvision.models.vgg import VGG16_Weights
from torchvision import models, transforms

import numpy as np
from tqdm import tqdm
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#####################################################################
#####################################################################
#apply function to image
def get_image(image_path, image_transformation, image_size=(300,300)):
    # Convert input to file-like object if necessary
    if isinstance(image_path, str):
        image_path = open(image_path, 'rb')
    elif isinstance(image_path, torch.Tensor):
        image_path = io.BytesIO(image_path.cpu().numpy())
        
    # Load image
    image_path.seek(0)
    image = Image.open(image_path)
    # Resize image
    image = ImageOps.exif_transpose(image)
    image = image.resize(image_size, resample=Image.Resampling.LANCZOS)
    # Apply transformation and add batch dimension
    image = image_transformation(image).unsqueeze(0)
    # Move image to device
    return image.to(device)


def gram_matrix(g_matrix):
    _, channels, height, width = g_matrix.size()
    # Reshape the tensor into a 2D tensor
    g_matrix = g_matrix.view(channels, height * width)
    # Get the matrix and multiply with its transpose
    g_matrix = torch.mm(g_matrix, g_matrix.t())
    return g_matrix

# after transform denormalize to be able to visualize the image
def denormalize(input_value):
    # in numpy number of channels is at the end
    # input size c x h x w
    input_value = input_value.numpy().transpose((1, 2, 0)) # CxHxW -> HxWxC
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input_value = std * input_value + mean
    input_value = np.clip(input_value, 0, 1)
    return input_value

#####################################################################
#####################################################################
# should use pre-trained network on images, here VGG will be used
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # define class variables
        self.selected_layers = ['3', '8', '15', '22', '27', '29']
        # define pre-trained model capture all layer
        # obtained layer values are all after ReLU activation function
        self.vgg = models.vgg16(weights=models.vgg.VGG16_Weights.IMAGENET1K_V1).features
        
    def forward(self, x):
        # loop over layers and save features of target layers
        layer_features = []
        # using modules parameters, convert model to a dictionary
        # looping of the tupel values
        for layer_number, layer in self.vgg._modules.items():
            x = layer(x)
            if layer_number in self.selected_layers:
                # appending target layers to the feature list
                layer_features.append(x)
        return layer_features
#####################################################################
#####################################################################
image_transformaion = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(   
                                                                mean = (0.485, 0.456, 0.406),
                                                                std=(0.229, 0.224, 0.225)
                                                                )
                                        ])

# lets load images
path = os.getcwd()
content = os.path.join(path, 'content2.jpg')
style = os.path.join(path, 'style.jpg')
# content image - original
content = get_image(content, image_transformaion)
content = content.to(device)
# style image to be learned
style = get_image(style, image_transformaion)
style = style.to(device)
 # or nn.Parameter(torch.FloatTensor(content_img.size()))
generated_image = content.clone()  
generated_image = generated_image.to(device)
generated_image.requires_grad = True 
# setting optimizer with generated image parameters
optimizer = torch.optim.Adam([generated_image], lr=1e-3, betas=[0.5, 0.999])
# initiate model
encoder = FeatureExtractor().to(device)
# pre-trained model no need to update weights
# or eval()
#for parameter in encoder.parameters():
#    parameter.requires_grad = False

#####################################################################
#####################################################################

content_weight = 1
style_weight = 1000

# set encoder to evaluation mode
encoder.eval()
# Create a tqdm instance and name it progress_bar
with tqdm(range(5000)) as progress_bar:  
    for epoch in progress_bar:
    
        content_features = encoder(content)
        style_features = encoder(style)
        generated_features = encoder(generated_image)
        
        content_loss = torch.mean((content_features[-1] - generated_features[-1])**2)  

        style_loss = 0
        for gf, sf in zip(generated_features, style_features):
            _, c, h, w = gf.size()
            gram_gf = gram_matrix(gf)
            gram_sf = gram_matrix(sf)
            style_loss += torch.mean((gram_gf - gram_sf)**2)  / (c * h * w) 

        loss = content_weight * content_loss + style_weight * style_loss 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            progress_bar.write('Epoch [{}]\tContent Loss: {:.4f}\tStyle Loss: {:.4f}'.format(epoch, content_loss.item(), style_loss.item()))
            progress_bar.set_description("Content Loss: {:.4f}, Style Loss: {:.4f}".format(content_loss.item(), style_loss.item()))

#####################################################################
#####################################################################
styled_image = generated_image.detach().cpu().squeeze()
styled_image = denormalize(styled_image)

plt.imshow(styled_image)
plt.grid(False)
plt.axis('off')
plt.show()
