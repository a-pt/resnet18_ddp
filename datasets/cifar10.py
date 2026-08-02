from torchvision import transforms
from torchvision.datasets import CIFAR10

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914,0.4822,0.4465),
        std=(0.2470,0.2435,0.2616)
    )
])  


test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914,0.4822,0.4465),
        std=(0.2470,0.2435,0.2616)
    )
])


def get_cifar10_datasets():
    train_dataset = CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=test_transform,
    )

    return train_dataset, test_dataset