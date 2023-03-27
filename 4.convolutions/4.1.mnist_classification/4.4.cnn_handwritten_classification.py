import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

from cnn_mnist_classification import CNN

#########################
# load the saved model
model = CNN()
model.load_state_dict(torch.load('cnn_model_weights.pth'))
model.eval()


# Load the image and apply transformations
image_path = r'4.convolutions\4.1.mnist_classification\image_files\2.png'
image = Image.open(image_path).convert('L')  # convert to grayscale
image = Image.eval(image, lambda x: 255 - x) # make background and writting colour same as training set
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