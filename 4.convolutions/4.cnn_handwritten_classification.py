import torch
from PIL import Image
import torchvision.transforms as transforms

from cnn_mnist_classification import CNN

# Load the trained model
model = torch.load('cnn_model.pth')

# Load the image and apply transformations
image_path = 'handwritten_digit.png'
image = Image.open(image_path).convert('L')  # convert to grayscale
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # resize to (28, 28)
    transforms.ToTensor(),  # convert to tensor
    transforms.Normalize((0.1307,), (0.3081,))  # normalize
])
image_tensor = transform(image).unsqueeze(0)  # add batch dimension

# Make predictions
model.eval()
with torch.no_grad():
    output = model(image_tensor)
    _, predicted = torch.max(output, 1)

# Print the predicted digit
print('The predicted digit is:', predicted.item())
