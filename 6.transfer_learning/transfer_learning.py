import os
import torch
import torchvision
import torch.optim
import torch.nn as nn

from torch.autograd import variable
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

import numpy as np
##########################################################
# use a powerful pretrained model i.e. AlexNet/ResNets?etc 
# 1) freez all layers
# 2) remove the last layer 
# 3) train and reconstruct last layer using your own data
#       -> image net has 1000 classes, your data has 5
#       -> change last layer to reflect 5
##########################################################
