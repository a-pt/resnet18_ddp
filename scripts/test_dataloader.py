import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from torch.utils.data import DataLoader
from datasets.cifar10 import get_cifar10_datasets


def main():

    # Load datasets
    train_dataset, test_dataset = get_cifar10_datasets()

    # Create DataLoader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=0,
    )

    # Fetch one batch
    images, labels = next(iter(train_loader))

    print("=" * 50)
    print("First Batch Information")
    print("=" * 50)

    print(f"Images Shape : {images.shape}")
    print(f"Labels Shape : {labels.shape}")

    print()

    print(f"Image dtype  : {images.dtype}")
    print(f"Label dtype  : {labels.dtype}")

    print()

    print(f"Image Min    : {images.min():.3f}")
    print(f"Image Max    : {images.max():.3f}")

    print()

    print("First 10 Labels:")
    print(labels[:10])

    print(f"Dataset Size      : {len(train_dataset)}")
    print(f"Number of Batches : {len(train_loader)}")

    print()

    print(f"Batch Size        : {images.size(0)}")
    print(f"Channels          : {images.size(1)}")
    print(f"Height            : {images.size(2)}")
    print(f"Width             : {images.size(3)}")


if __name__ == "__main__":
    main()