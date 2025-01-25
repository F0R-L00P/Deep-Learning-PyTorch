import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

###############################################################################
#                                DEVICE SETUP
###############################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

###############################################################################
#                             DATA PREPARATION
###############################################################################
transform = transforms.Compose(
    [
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # Normalize to [-1, 1]
    ]
)

dataset = CelebA(root="data", split="train", transform=transform, download=True)
print(f"Dataset size: {len(dataset)} images")

dataloader = DataLoader(
    dataset,
    batch_size=128,  # Must be > 1 if using BatchNorm
    shuffle=True,
    num_workers=0,
)


###############################################################################
#                            GENERATOR NETWORK
###############################################################################
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512 * 4 * 4),
            nn.BatchNorm1d(512 * 4 * 4),
            nn.ReLU(True),
            nn.Unflatten(1, (512, 4, 4)),  # (B, 512, 4, 4)
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),  # Output in range [-1, 1]
        )

    def forward(self, x):
        return self.model(x)


###############################################################################
#                           DISCRIMINATOR NETWORK
###############################################################################
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


###############################################################################
#                               INSTANTIATE MODELS
###############################################################################
latent_dim = 100
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

# Test with a small batch
test_input_for_discriminator = torch.randn(8, 3, 64, 64).to(device)
print("Discriminator output shape:", discriminator(test_input_for_discriminator).shape)

test_input_for_generator = torch.randn(8, latent_dim).to(device)
print("Generator output shape:", generator(test_input_for_generator).shape)

###############################################################################
#                            LOSS AND OPTIMIZERS
###############################################################################
criterion = nn.BCELoss()
discriminator_optimizer = torch.optim.Adam(
    discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)
)
generator_optimizer = torch.optim.Adam(
    generator.parameters(), lr=0.0002, betas=(0.5, 0.999)
)


###############################################################################
#                       FAKE IMAGE SAVING FUNCTION
###############################################################################
def save_fake_images(generator, epoch, latent_dim, device, save_dir="generated_images"):
    os.makedirs(save_dir, exist_ok=True)

    latent_vector = torch.randn(16, latent_dim, device=device)  # Generate 16 images
    fake_images = generator(latent_vector).detach().cpu()

    # Denormalize
    fake_images = fake_images * 0.5 + 0.5
    grid = torchvision.utils.make_grid(fake_images, nrow=4)

    save_path = os.path.join(save_dir, f"epoch_{epoch+1}.png")
    torchvision.utils.save_image(grid, save_path)

    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis("off")
    plt.title(f"Epoch {epoch+1}")
    plt.show()


###############################################################################
#                               TRAINING LOOP
###############################################################################
num_epochs = 10
real_label = 1.0
fake_label = 0.0

# 1) Lists to store D and G losses for visualization
d_losses = []
g_losses = []

print("Starting Training...")

for epoch in range(num_epochs):
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

    for i, (real_images, _) in enumerate(progress_bar):
        real_images = real_images.to(device)
        batch_size = real_images.size(0)

        # Labels
        real_labels = torch.full((batch_size,), real_label, device=device)
        fake_labels = torch.full((batch_size,), fake_label, device=device)

        # ---------------------
        # 1) Train the Discriminator
        # ---------------------
        discriminator_optimizer.zero_grad()

        # a) Real images
        real_outputs = discriminator(real_images).view(-1)  # D(x)
        loss_real = criterion(real_outputs, real_labels)

        # b) Fake images
        latent_vector = torch.randn(
            batch_size, latent_dim, device=device
        )  # random gaussian distribution
        fake_images = generator(latent_vector)
        fake_outputs = discriminator(fake_images.detach()).view(-1)  # D(G(z))
        loss_fake = criterion(fake_outputs, fake_labels)

        # c) Combine & update Discriminator
        d_loss = loss_real + loss_fake
        d_loss.backward()
        discriminator_optimizer.step()

        # ---------------------
        # 2) Train the Generator
        # ---------------------
        generator_optimizer.zero_grad()

        fake_outputs_for_generator = discriminator(fake_images).view(-1)
        g_loss = criterion(fake_outputs_for_generator, real_labels)  # want to fool D

        g_loss.backward()
        generator_optimizer.step()

        # ---------------------
        # 3) Logging
        # ---------------------
        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())

        real_score = real_outputs.mean().item()  # Average D(x)
        fake_score = fake_outputs.mean().item()  # Average D(G(z))

        progress_bar.set_postfix(
            {
                "D Loss": f"{d_loss.item():.4f}",
                "G Loss": f"{g_loss.item():.4f}",
                "D(x)": f"{real_score:.2f}",
                "D(G(z))": f"{fake_score:.2f}",
            }
        )

    save_fake_images(generator, epoch, latent_dim, device)

print("Training Completed!")

###############################################################################
#                               PLOT LOSSES
###############################################################################
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(g_losses, label="G Loss")
plt.plot(d_losses, label="D Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
