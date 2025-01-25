import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

##############################################################################
#                          Hyperparameters & Setup
##############################################################################
image_size = 784  # 28 x 28
hidden_size = 256
latent_size = 64
num_epochs = 50
batch_size = 100
learning_rate = 0.0002

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create output folder to save generated images
os.makedirs("gan_images", exist_ok=True)

##############################################################################
#                       Datasets & Dataloaders
##############################################################################
# We normalize the MNIST images to [-1,1] so that they match Tanh outputs
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

train_dataset = dsets.MNIST(
    root="./data", train=True, transform=transform, download=True
)

data_loader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
)
# drop_last=True avoids partial batches completely

##############################################################################
#                       Discriminator & Generator
##############################################################################
# Discriminator
D = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(hidden_size, 1),
    nn.Sigmoid(),
).to(device)

# Generator
G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(inplace=True),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(inplace=True),
    nn.Linear(hidden_size, image_size),
    nn.Tanh(),
).to(device)

##############################################################################
#                       Loss and Optimizers
##############################################################################
criterion = nn.BCELoss()
d_optimizer = optim.Adam(D.parameters(), lr=learning_rate, betas=(0.5, 0.999))
g_optimizer = optim.Adam(G.parameters(), lr=learning_rate, betas=(0.5, 0.999))


##############################################################################
#                       Helper Functions
##############################################################################
def denorm(x):
    """
    Convert [-1,1] range tensor to [0,1] for visualization.
    """
    out = (x + 1) / 2
    return out.clamp(0, 1)


def save_fake_images(epoch):
    """
    Generate 10 sample images from the Generator and save them.
    """
    z = torch.randn(10, latent_size).to(device)
    fake_images = G(z)
    fake_images = fake_images.view(fake_images.size(0), 1, 28, 28)
    fake_images = denorm(fake_images)

    grid = torchvision.utils.make_grid(fake_images, nrow=10)
    ndarr = grid.mul(255).byte().cpu().numpy().transpose((1, 2, 0))
    plt.imshow(ndarr.squeeze(), cmap="gray")
    plt.axis("off")
    plt.savefig(f"gan_images/epoch_{epoch:03d}.png")
    plt.close()


##############################################################################
#                       Training Loop
##############################################################################
for epoch in range(num_epochs):
    for batch_idx, (images, _) in enumerate(
        tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    ):

        # Flatten images: (batch_size, 1, 28, 28) -> (batch_size, 784)
        images = images.view(images.size(0), -1).to(device)

        # Real and fake labels, one and zero respectively
        real_labels = torch.ones(images.size(0), 1).to(device)
        fake_labels = torch.zeros(images.size(0), 1).to(device)

        # -----------------------------------------------------
        # Train the Discriminator
        # -----------------------------------------------------
        # Forward real images
        real_outputs = D(images)
        d_loss_real = criterion(real_outputs, real_labels)

        # Forward fake images
        z = torch.randn(images.size(0), latent_size).to(
            device
        )  # z-> noise, sampled from random normal (gaussian) distribution
        fake_images = G(z)
        fake_outputs = D(fake_images)
        d_loss_fake = criterion(fake_outputs, fake_labels)

        # Backprop + Optimize
        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # -----------------------------------------------------
        # Train the Generator
        # -----------------------------------------------------
        z = torch.randn(images.size(0), latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        g_loss = criterion(
            outputs, real_labels
        )  # we want the generator to produce images D thinks are real

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

    # Logging
    print(
        f"Epoch [{epoch+1}/{num_epochs}], "
        f"d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}, "
        f"D(x): {real_outputs.mean().item():.2f}, D(G(z)): {fake_outputs.mean().item():.2f}"
    )

    # Save sample fake images
    save_fake_images(epoch + 1)

##############################################################################
#                       Visualize Final Samples
##############################################################################
# Generate and plot some final images
z = torch.randn(batch_size, latent_size).to(device)
fake_images = G(z).view(batch_size, 1, 28, 28)
fake_images = denorm(fake_images)

fig, axes = plt.subplots(1, 10, figsize=(15, 2))
for i in range(10):
    axes[i].imshow(fake_images[i][0].cpu().detach().numpy(), cmap="gray")
    axes[i].axis("off")
plt.show()
