from sys import platform
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from PIL import Image
import torch
import config


if platform == "darwin":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") 
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

labels = ["Cat", "Dog"]
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])

model = models.resnet18(weights=None)
for param in model.parameters():
    param.requires_grad = False
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, config.NUM_CLASSES)
model.to(device)

checkpoint = torch.load(config.PATH_SAVE_MODEL, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

image = Image.open(config.PATH_IMAGE_PREDICT)
image = transform(image)
image = torch.unsqueeze(image, 0)

with torch.no_grad():
    outputs = model(image.to(device))

output_label = torch.topk(outputs, 1)
pred_class = labels[int(output_label.indices)]

print("Prediction: ", pred_class)

