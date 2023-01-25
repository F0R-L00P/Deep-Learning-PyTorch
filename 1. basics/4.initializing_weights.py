import torch
import torch.nn as nn

# Define a tensor to initalize
# define a layer and access its weights
layer = nn.Linear(8, 8)
# viewing weights
layer.weight
# if only need to see weights without any information use
layer.weight.data
# accessing weights at the first layer
layer.weight[0]
# without info
layer.weight[0].data

# initialize the weight tensor giving it a uniform distribution
# giving it a minimum of 0, maximum of 3
nn.init.uniform_(layer.weight.data, a=0.0, b=3)

# lets try the normal distribution (gaussian) 
# and initialize weights based on normal, can change mean and std
# can set any range distribution 
# https://www.google.com/url?sa=i&url=https%3A%2F%2Fsites.nicholas.duke.edu%2Fstatsreview%2Fcontinuous-probability-distributions%2F&psig=AOvVaw2F5PM0VlFh7xdHXWheDJ-h&ust=1674572930063000&source=images&cd=vfe&ved=0CBEQjhxqFwoTCOip_bf83fwCFQAAAAAdAAAAABAE
nn.init.normal_(layer.weight, mean=0, std=1)
nn.init.normal_(layer.weight, mean=0, std=0.2)

# can also set bias value if needed to a specific constant
nn.init.constant_(layer.bias, 8)
layer.bias