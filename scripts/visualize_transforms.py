import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import torch

from datasets.cifar10 import get_cifar10_datasets

train_dataset, test_dataset = get_cifar10_datasets()


def denormalize(image):
    """
    Undo Normalize() so matplotlib can display the image correctly.
    """
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)

    image = image * std + mean
    image = image.clamp(0, 1)

    return image


fig, axes = plt.subplots(1, 5, figsize=(15, 3))
index = 0

for i in range(5):

    image, label = train_dataset[index]
    image = denormalize(image)

    # Convert CHW -> HWC
    image = image.permute(1, 2, 0)

    axes[i].imshow(image)
    axes[i].set_title(f"Version {i+1}")
    axes[i].axis("off")

plt.suptitle("Five Augmented Versions of the Same Image")
plt.tight_layout()

plt.show()