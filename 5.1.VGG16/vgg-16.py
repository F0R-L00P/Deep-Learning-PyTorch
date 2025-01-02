# bsic libraries
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output

from PIL import Image

# DL libraries
import torch
import torch.nn
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader

# designed libraries
from helper_functions import train, validate, set_seed

# %matplotlib inline

#######################
###### Settings#########
#######################
random_seed = 1818
batch_size = 32
num_epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available.")

# Deterministic seed
set_seed(random_seed)

#######################
# Using CIFAR-10 dataset
#######################
train_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomCrop(32, padding=4),  # common CIFAR-10 practice
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        ),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

test_transform = torchvision.transforms.Compose(
    [
        #        torchvision.transforms.CenterCrop((32, 32)),  # Keep size consistent
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

train_set = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=train_transform,
)

test_set = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=test_transform,
)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)

test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

# Inspect DataLoader
print(f"Training set size: {len(train_set)}")
print(f"Test set size: {len(test_set)}")
print(f"Number of batches in train_loader: {len(train_loader)}")
print(f"Number of batches in test_loader: {len(test_loader)}")


########################
# VGG-16 Architecture
########################
class VGG16(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.features = torch.nn.Sequential(
            # BLOCK 1
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # BLOCK 2
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # BLOCK 3
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # BLOCK 4
            torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # BLOCK 5
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Fully connected layers
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512, 4096),  # Adjusted input size
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        # print(f"Shape after features: {x.shape}")  # Debugging line
        x = torch.flatten(x, 1)  # Flatten the feature map
        # print(f"Shape after flatten: {x.shape}")  # Debugging line
        x = self.classifier(x)
        return x


###################### TEST
# Initialize the model
# model = VGG16(num_classes=10).to(device)

# Generate random input
# random_input = torch.randn(1, 3, 32, 32).to(device)

# Forward pass
# output = model(random_input)

# Check the output
# print("Output shape:", output.shape)  # Should be [1, 10]

#########################################
########### Training the model ##########
#########################################
# Define the model, loss, and optimizer
model = VGG16(num_classes=10).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Lists to store the history of accuracies and losses
train_acc_history = []
val_acc_history = []
train_loss_history = []
val_loss_history = []

best_accuracy = 0.0

for epoch in range(num_epochs):
    # Clear the output at the start of the epoch to have a "live" effect
    #    clear_output(wait=True)
    print(f"Epoch {epoch + 1}/{num_epochs}")

    # Train and validate
    train_loss, train_accuracy = train(
        model, train_loader, optimizer, criterion, device
    )
    val_loss, val_accuracy = validate(model, test_loader, criterion, device)

    # Store the metrics
    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)
    train_acc_history.append(train_accuracy)
    val_acc_history.append(val_accuracy)

    # Print metrics to the terminal
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    # Plot the metrics
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label="Train Loss", color="blue")
    plt.plot(val_loss_history, label="Val Loss", color="orange")
    plt.title("Loss vs. Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history, label="Train Acc", color="blue")
    plt.plot(val_acc_history, label="Val Acc", color="orange")
    plt.title("Accuracy vs. Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.tight_layout()
    plt.show()

test_loss, test_accuracy = validate(model, test_loader, criterion, device)
print(f"Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
####################################################
################visual inspection##################
####################################################
# Get one batch of images and their labels from the test loader
inputs, labels = next(iter(test_loader))
inputs, labels = inputs.to(device), labels.to(device)

# Pass them through the model
model.eval()
with torch.no_grad():
    outputs = model(inputs)
_, preds = torch.max(outputs, 1)

classes = test_set.classes  # CIFAR-10 class names

# Set up a 4x8 grid for plotting 32 images
fig, axes = plt.subplots(4, 8, figsize=(20, 10))
axes = axes.flatten()  # Flatten the array of axes for easy indexing

# We'll show the first 32 images from the batch
for idx in range(32):
    img = inputs[idx].cpu().numpy().transpose((1, 2, 0))
    # Unnormalize the image (since we used mean=0.5, std=0.5)
    img = img * 0.5 + 0.5
    img = np.clip(img, 0, 1)  # Ensuring pixel values are between 0 and 1

    axes[idx].imshow(img)
    axes[idx].set_title(f"Pred: {classes[preds[idx]]}\nLabel: {classes[labels[idx]]}")
    axes[idx].axis("off")

plt.tight_layout()
plt.show()

##########################################################
###############confusion matrix&F1 score#################
##########################################################
from sklearn.metrics import confusion_matrix, classification_report

# make sure model weights are locked
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(cm)

# Classification report
print("Classification Report:")
print(classification_report(all_labels, all_preds))
####################################################################
####################################################################
# Create a checkpoint dictionary
checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "best_accuracy": best_accuracy,
    # 'scheduler_state_dict': scheduler.state_dict()
}

# Save the checkpoint
torch.save(checkpoint, "vgg16_checkpoint.pth")
