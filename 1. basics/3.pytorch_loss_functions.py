import torch
import torch.nn as nn
######################################
# lets learn how to use LOSS functions
######################################
# making prediction tensor 4 features with 5 outputs
prediction = torch.randn(4, 5)
label = torch.randn(4, 5)

mse = torch.nn.MSELoss(reduction='mean')

# lets calcualte the loss for the input and target
# output is subtracted from eachother and squared the result 
# NOTE: single value must be used to compute the loss. 
    # lets calculate the mean, as an example
    # parameter reduction can be set to have the desired operation
loss = mse(prediction, label)
print(loss)

# this can be manually computed via
((prediction - label)**2).mean()

# let generate random labels, drawn from uniform distribution
label = torch.zeros(4, 5).random_(0, 2)
print(label)

# lets define a BCE loss, passing through the sigmoid layer
sigmoid = nn.Sigmoid()
bianry_loss = nn.BCELoss(reduction='mean')

# lets compute the loss, when passed via the non-linear layer
bianry_loss(sigmoid(prediction), label)

# if you need logit loss, bypass sigmoind
bianry_loss_logits = nn.BCEWithLogitsLoss(reduction='mean')
bianry_loss_logits(prediction, label)