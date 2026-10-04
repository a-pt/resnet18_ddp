from dataclasses import dataclass

@dataclass
class Config:
    # Dataset
    dataset: str = "CIFAR10"

    # Training
    batch_size: int = 128
    epochs: int = 10
    learning_rate: float = 1e-3

    # Device
    device: str = "cuda"

    # Misc
    seed: int = 42
    num_workers: int = 0