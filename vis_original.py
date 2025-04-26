import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms

# Define transformation for normalizing the data
transform = transforms.Compose([transforms.ToTensor()])

# Load CIFAR-100 dataset
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=False, transform=transform)

# Get class labels
class_labels = trainset.classes

# Get the index for 'keyboard' class
keyboard_class_index = class_labels.index('apple')

# Collect all images of the 'keyboard' class
keyboard_images = []
for img, label in trainset:
    if label == keyboard_class_index:
        keyboard_images.append(img)

# Plot the 'keyboard' images in a grid
n_images = len(keyboard_images)
grid_size = int(n_images ** 0.5) + 1  # Calculate grid size for visualization

fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
axes = axes.flatten()

for i, ax in enumerate(axes):
    if i < n_images:
        ax.imshow(keyboard_images[i].permute(1, 2, 0))  # Convert from CxHxW to HxWxC for displaying
        ax.axis('off')
    else:
        ax.axis('off')

fig.suptitle("All 'Keyboard' Images from CIFAR-100", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()
