import os
import torch
import torchvision
import torch.nn as nn

from torch.autograd import Variable
from torchvision import datasets, models, transforms

import json
import scipy.misc
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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