import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image

from tqdm import tqdm
###############################################################
###############################################################
# get the current working directory
path = os.getcwd()

train_dataset = torchvision.datasets.SVHN(root=path,
                                           split='train',
                                           transform=transforms.ToTensor(),
                                           download=True
                                        )
test_dataset = torchvision.datasets.SVHN(root=path,
                                           split='test',
                                           transform=transforms.ToTensor(),
                                           download=True
                                        )

batch_size = 64

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True
                                        )
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=True
                                        )

# creating directory to store sample output images
results_directory = 'results2'
if not os.path.exists(results_directory):
    os.makedirs(results_directory)

###################################################################
###################################################################
# flattened mnist image
image_size = 3*32*32    
#hidden layer
hidden_dim=400
# laten space dim
latent_dim = 20

# VAE Model Architecture
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # first fully connected layer
        self.fc1 = nn.Linear(in_features=image_size, out_features=hidden_dim)
        # mapping to hidden layer for mean
        self.fc2_mean = nn.Linear(in_features=hidden_dim, out_features=latent_dim)
        # mapping to hidden layer for std
        self.fc2_logvar = nn.Linear(in_features=hidden_dim, out_features=latent_dim)
        # transition to the decod layer after mean and logvar
        self.fc3 = nn.Linear(in_features=latent_dim, out_features=hidden_dim)
        # move to final output layer to generate representation
        self.fc4 = nn.Linear(in_features=hidden_dim, out_features=image_size)

    # The encode() method represents the encoder part of the VAE
    # encoder takes an input tensor x and maps it to the latent space, after hidden layer
    # returning two parameters: the mean (mu) and the log variance (log_var)
    def encode_hidden(self, x):
        # Pass the input tensor x through the first fully connected layer self.fc1 and apply
        # the ReLU activation function. Return the resulting tensor.
        return F.relu(self.fc1(x))

    def encode_mean(self, hidden_output):
        # Pass the hidden representation hidden_output through the fully connected layer self.fc2_mean
        # to compute the mean mu of the approximate posterior distribution of the latent variables.
        # Return the resulting tensor.
        return self.fc2_mean(hidden_output)
    
    def encode_logvar(self, hidden_output):
        # Pass the hidden representation hidden_output through the fully connected layer self.fc2_logvar
        # to compute the log variance log_var of the approximate posterior distribution of the
        # latent variables. Return the resulting tensor.
        return self.fc2_logvar(hidden_output)
    
    # calling all encod methods
    def encode(self, x):
        # Compute the hidden representation hidden_output by calling the 
        # encode_hidden() method with input tensor x.
        hidden_output = self.encode_hidden(x)
        # Calculate the mean mu of the approximate posterior distribution of the latent variables
        # by calling the encode_mean() method with the hidden representation hidden_output.
        mu = self.encode_mean(hidden_output)
        # Calculate the log variance log_var of the approximate posterior distribution of the latent
        # variables by calling the encode_logvar() method with the hidden representation hidden_output.
        log_var = self.encode_logvar(hidden_output)
        # Return the mean mu and log variance log_var.
        return mu, log_var
    
    def reparameterize(self, mu, logvar):
        # Compute the standard deviation by exponentiating the 
        # log variance and taking the square root.
        std = torch.exp(logvar / 2)
        # Sample a random noise tensor from the standard normal distribution 
        # with the same shape as std.
        eps = torch.randn_like(std)
        # Add the mean mu and the element-wise product of std and eps 
        # to obtain the latent variable z.
        return mu + std * eps
    
    def decode(self, z):
        hidden_output = F.relu(self.fc3(z))
        out = torch.sigmoid(self.fc4(hidden_output))
        return out
    
    def forward(self, x):
        # Encode the input tensor x into the mean mu and log variance log_var.
        mu, logvar = self.encode(x.view(-1, image_size))
        # Reparameterize the distribution using the mean mu and log variance log_var 
        # to obtain the latent variable z.
        z = self.reparameterize(mu, logvar)
        # Decode the latent variable z into the reconstructed output.
        reconstructed = self.decode(z)
        # Return the reconstructed output, mean mu, and log variance log_var.
        return reconstructed, mu, logvar
    
###################################################################
###################################################################
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#initialize model
vae_model = VAE().to(device)
# settingup optimizer
optimizer = torch.optim.Adam(vae_model.parameters(), lr=1e-3)

# desiging a loss function
# The VAE's overall loss function is the sum of the reconstruction loss 
# and the KL divergence, weighted by a hyperparameter called the beta.  
# VAE_Loss = Reconstruction_Loss + beta * KL_Divergence
def vae_loss(reconstructed, original, mu, logvar, beta=1.0):
    # Reconstruction Loss (Binary Cross Entropy)
    reconstruction_loss = F.binary_cross_entropy(reconstructed.view(-1, 3, 32, 32), original, reduction='sum')
    # KL Divergence
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Total VAE Loss
    loss = reconstruction_loss + beta * kl_divergence
    # return loss
    return loss


# define training function to pass over data encode images to the latent space

def training_stage(epoch):
    vae_model.train()
    train_loss = 0
    for i, (images, _) in enumerate(tqdm(train_loader)):
        images = images.to(device)
        # obtain forward pass values
        reconstructed, mu, logvar = vae_model(images)
        # Calculate the VAE loss
        loss = vae_loss(reconstructed, images, mu, logvar)
        # zero gradient prevent transfer between batches
        optimizer.zero_grad()
        # Backpropagation
        loss.backward()
        train_loss += loss.item()
        # Update weights
        optimizer.step()

    print('=====> Epoch {}, Average Train Loss: {:.3f}'.format(epoch, train_loss / len(train_loader)))

# Test function
def testing_stage(epoch):
    # lock model weights
    vae_model.eval()
    test_loss = 0
    # do not update gradient
    with torch.no_grad():
        for batch_idx, (images, _) in tqdm(enumerate(test_loader), total=len(test_loader)):
            # pass image to GPU
            images = images.to(device)
            # obtain forward pass values
            reconstructed, mu, logvar = vae_model(images)
            # calculate loss
            test_loss += vae_loss(reconstructed, images, mu, logvar).item()
            if batch_idx == 0:
                comparison = torch.cat([images[:5], reconstructed.view(batch_size, 3, 32, 32)[:5]])
                save_image(comparison.cpu(), 'results2/reconstruction_' + str(epoch) + '.png', nrow=5)

    print('=====> Average Test Loss: {:.3f}'.format(test_loss / len(test_loader)))

###################################################################
###################################################################
# lets loop over the data, encode-decode train model and obtain predictions to view
# Main function
epochs = 10
for epoch in range(1, epochs + 1):
    training_stage(epoch)
    testing_stage(epoch)
    with torch.no_grad():
        # Get rid of the encoder and sample z from the gaussian distribution 
        # and feed it to the decoder to generate samples
        sample = torch.randn(64, 20).to(device)
        generated = vae_model.decode(sample).cpu()
        # Resize the images to [3, 32, 32] and save them
        generated_resized = generated.view(64, 3, 32, 32)
        save_image(generated_resized, 'results2/sample_' + str(epoch) + '.png')




