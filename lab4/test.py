import torch
import torchvision

dataset = torchvision.datasets.CIFAR10("data", download=True, train=True)
img, label = dataset[1]
print(type(img))
print(label)
img
