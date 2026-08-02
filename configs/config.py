from dataclasses import dataclass

@dataclass
class Config:
    # Dataset
    dataset = "CIFAR10"

    # Training
    batch_size = 128
    epochs = 20
    learning_rate = 1e-3

    # Device
    device = "cuda"

    # Misc
    seed = 42
    num_workers = 4