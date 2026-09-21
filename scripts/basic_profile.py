import torch
from torch.profiler import profile, ProfilerActivity

from torch.utils.data import DataLoader

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from datasets.cifar10 import get_cifar10_datasets
from models.resnet18 import ResNet18


BATCH_SIZE = 128

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    train_dataset, _ = get_cifar10_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    model = ResNet18(num_classes=10)
    model.to(device)
    model.train()

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=3,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            "./runs/profile"
        ),
    ) as prof:

        for step, (images, labels) in enumerate(train_loader):

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()

            optimizer.step()
            prof.step()

            if step >= 4:
                break

        print(
            prof.key_averages().table(
                sort_by="cuda_time_total",
                row_limit=20,
            )
        )

if __name__ == "__main__":
    main()