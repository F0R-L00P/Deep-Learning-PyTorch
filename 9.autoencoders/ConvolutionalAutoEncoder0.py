import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

###############################################################################
# Device
###############################################################################
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

###############################################################################
# Hyperparameters
###############################################################################
RANDOM_SEED = 123
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 10
NUM_CLASSES = 10  # For MNIST (digits 0–9)

torch.manual_seed(RANDOM_SEED)

###############################################################################
# Data Loading
###############################################################################
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        # If you want to normalize to [0,1], you can omit normalization
        # transforms.Normalize((0.1307,), (0.3081,))
    ]
)

train_dataset = datasets.MNIST(
    root="data", train=True, transform=transform, download=True
)

test_dataset = datasets.MNIST(
    root="data", train=False, transform=transform, download=True
)

# For demonstration, we'll treat part of the training set as "validation"
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

###############################################################################
# Quick Data Check
###############################################################################
print("Training Set:\n")
train_iter = iter(train_loader)
images, labels = next(train_iter)
print("Image batch dimensions:", images.size())
print("Image label dimensions:", labels.size())
print(labels[:10])

print("\nValidation Set:")
valid_iter = iter(valid_loader)
images, labels = next(valid_iter)
print("Image batch dimensions:", images.size())
print("Image label dimensions:", labels.size())
print(labels[:10])

print("\nTesting Set:")
test_iter = iter(test_loader)
images, labels = next(test_iter)
print("Image batch dimensions:", images.size())
print("Image label dimensions:", labels.size())
print(labels[:10])


###############################################################################
# Model Definition
###############################################################################
class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Trim(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Trims any extra row/col to ensure 28x28 output
        return x[:, :, :28, :28]


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, stride=1, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 64, stride=2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, stride=2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, stride=1, kernel_size=3, padding=1),
            nn.Flatten(),
            nn.Linear(3136, 4),
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 3136),
            Reshape(-1, 64, 7, 7),
            nn.ConvTranspose2d(64, 64, stride=1, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 64, stride=2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 32, stride=2, kernel_size=3, padding=0),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(32, 1, stride=1, kernel_size=3, padding=0),
            Trim(),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


###############################################################################
# Create Model, Optimizer, Loss
###############################################################################
model = AutoEncoder().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCELoss()


###############################################################################
# Helper Plotting Functions (No List Comprehensions)
###############################################################################
def plot_training_loss(train_loss_list, num_epochs):
    plt.figure()
    plt.title("Training Loss Over Batches")
    x_values = range(len(train_loss_list))  # no list comprehension
    plt.plot(x_values, train_loss_list, color="blue")
    plt.xlabel("Batch Index")
    plt.ylabel("Loss")
    plt.grid(True)
    # We can also show lines for epoch boundaries
    if num_epochs > 1:
        batch_per_epoch = len(train_loader)
        for epoch_i in range(1, num_epochs):
            plt.axvline(
                x=epoch_i * batch_per_epoch, color="red", linestyle="--", alpha=0.5
            )
    plt.tight_layout()


def plot_generated_images(data_loader, model, device):
    plt.figure(figsize=(8, 2))
    plt.suptitle("Original vs Reconstructed Images", y=1.05)

    # Take a small batch
    small_iter = iter(data_loader)
    small_images, _ = next(small_iter)
    small_images = small_images.to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        reconstructed = model(small_images)

    # Move to CPU for plotting
    small_images = small_images.cpu()
    reconstructed = reconstructed.cpu()

    # Plot first 8 images
    n_plots = 8
    for i in range(n_plots):
        plt.subplot(2, n_plots, i + 1)
        plt.imshow(small_images[i][0].numpy(), cmap="binary")
        plt.axis("off")

        plt.subplot(2, n_plots, n_plots + i + 1)
        plt.imshow(reconstructed[i][0].numpy(), cmap="binary")
        plt.axis("off")
    plt.tight_layout()


def plot_latent_space_with_labels(num_classes, data_loader, model, device):
    model.eval()
    encoded_list = []
    label_list = []

    # Gather latent vectors and labels
    for batch_i, (features, targets) in enumerate(data_loader):
        features = features.to(device)
        with torch.no_grad():
            encoded_batch = model.encoder(features).cpu().numpy()
        targets = targets.cpu().numpy()

        # Append without list comprehensions
        for item in encoded_batch:
            encoded_list.append(item)
        for item in targets:
            label_list.append(item)

    encoded_array = np.array(encoded_list)
    label_array = np.array(label_list)

    plt.figure()
    plt.title("Latent Space")
    # Plot each class separately
    for c in range(num_classes):
        points_x = []
        points_y = []
        for j in range(len(label_array)):
            if label_array[j] == c:
                points_x.append(encoded_array[j, 0])
                points_y.append(encoded_array[j, 1])
        plt.scatter(points_x, points_y, label=str(c), alpha=0.5)
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.grid(True)
    plt.tight_layout()


###############################################################################
# Training Loop
###############################################################################
log_dict = {"train_loss_per_batch": []}

model.train()
for epoch in range(NUM_EPOCHS):
    for batch_idx, (features, _) in enumerate(train_loader):
        features = features.to(DEVICE)

        # Forward pass
        decoded = model(features)
        loss = criterion(decoded, features)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        log_dict["train_loss_per_batch"].append(loss.item())

        # Print progress every 250 batches
        if (batch_idx + 1) % 250 == 0:
            print(
                f"Epoch: {epoch + 1}/{NUM_EPOCHS} "
                f"| Batch: {batch_idx + 1}/{len(train_loader)} "
                f"| Loss: {loss.item():.4f}"
            )

print("\nTraining complete.")

###############################################################################
# Post-Training Visualizations
###############################################################################
plot_training_loss(log_dict["train_loss_per_batch"], NUM_EPOCHS)
plt.show()

plot_generated_images(data_loader=train_loader, model=model, device=DEVICE)

plot_latent_space_with_labels(
    num_classes=NUM_CLASSES, data_loader=train_loader, model=model, device=DEVICE
)
plt.legend()
plt.show()

# Example of manually decoding a latent vector
with torch.no_grad():
    # Create a 4D latent vector on the GPU device
    manual_latent = torch.tensor([2.5, -2.5, 1.0, -1.0], device=DEVICE)
    manual_latent = manual_latent.unsqueeze(0)  # shape => [1, 4]

    new_image = model.decoder(manual_latent)
    new_image = new_image.squeeze(0).squeeze(0)

plt.figure()
plt.title("Manually Decoded Image from Latent [2.5, -2.5, 1.0, -1.0]")
plt.imshow(new_image.cpu().numpy(), cmap="binary")
plt.axis("off")
plt.show()
