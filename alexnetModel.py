import torch, torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
import time

import numpy as np
import matplotlib.pyplot as plt
import os

from PIL import Image

# Applying Transforms to the Data
image_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

alexnet = models.alexnet(pretrained=True)
alexnet

# Freeze model parameters
for param in alexnet.parameters():
    param.requires_grad = False


num_classes = 1
idx_to_class = {0: 'cattle'}

# Change the final layer of AlexNet Model for Transfer Learning
alexnet.classifier[6] = nn.Linear(4096, num_classes)
alexnet.classifier.add_module("7", nn.LogSoftmax(dim = 1))
alexnet

# Define Optimizer and Loss Function
loss_func = nn.NLLLoss()
optimizer = optim.Adam(alexnet.parameters())


alexnet.load_state_dict(torch.load('my_model.pth', map_location=torch.device('cpu')))

# def predict(test_image_name, model=alexnet):

#     '''
#     Function to predict the class of a single test image
#     Parameters
#         :param model: Model to test
#         :param test_image_name: Test image
#     '''

#     transform = image_transforms['test']

#     # test_image = Image.open(test_image_name)
#     # plt.imshow(test_image)

#     test_image_tensor = transform(test_image)

#     # Move the model to the CPU if it's currently on GPU
#     model.to('cpu')

#     # Move the input tensor to the CPU
#     test_image_tensor = test_image_tensor.view(1, 3, 224, 224).cpu()

#     with torch.no_grad():
#         model.eval()
#         # Model outputs log probabilities
#         out = model(test_image_tensor)
#         ps = torch.exp(out)
#         # print(out)
        
#         # Move tensors to CPU before converting to numpy
#         topclass_cpu = out.cpu().numpy()
#         topk_cpu = ps.cpu().numpy()

#         for i in range(1):
#             print("Prediction", i+1, ":", idx_to_class[int(topclass_cpu[0][i])])

# import cv2
# test_image = cv2.imread(r'C:\Users\moaaz\Desktop\project\runs\detect\predict2\image0.jpg')
# predict(test_image)

import torch
import cv2
from torchvision import transforms
from PIL import Image

def predict(test_image, model=alexnet):
    '''
    Function to predict the class of a single test image using OpenCV
    Parameters:
        - model: Model to test
        - test_image_path: Path to the test image
    '''
    transform = image_transforms['test']

    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)  # Convert to RGB format

    # Apply the same transformation used for the PIL image
    pil_image = transforms.ToPILImage()(test_image)
    test_image_tensor = transform(pil_image)

    # Move the model to the CPU if it's currently on GPU
    model.to('cpu')

    # Move the input tensor to the CPU
    test_image_tensor = test_image_tensor.view(1, 3, 224, 224).cpu()

    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        out = model(test_image_tensor)
        ps = torch.exp(out)

        # Move tensors to CPU before converting to numpy
        topclass_cpu = out.cpu().numpy()
        topk_cpu = ps.cpu().numpy()

        for i in range(1):
            return f"The image contains {idx_to_class[int(topclass_cpu[0][i])]}"

# Example usage
# test_image_path = r'C:\Users\moaaz\Desktop\project\runs\detect\predict2\image0.jpg'
# test_image = cv2.imread(test_image_path)
# predict(test_image)