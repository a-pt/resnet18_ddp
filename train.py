import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.checkpoint import save_checkpoint
from configs.config import Config
from utils.device import get_device
from utils.seed import set_seed
from datasets.cifar10 import get_cifar10_datasets
from models.resnet18 import ResNet18

from engine import train_one_epoch, validate


def main():
    config = Config()
    set_seed(config.seed)
    device = get_device()

    writer = SummaryWriter(log_dir="runs/resnet18")

    print("=" * 50)
    print("Configuration")
    print("=" * 50)
    print(config)
    print(f"\nRunning on: {device}")

    train_dataset, test_dataset = get_cifar10_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    images, labels = next(iter(train_loader))

    writer.add_images("Training Images",images[:16])

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = ResNet18(num_classes=10)
    model.to(device)

    dummy = torch.randn(1, 3, 32, 32).to(device)
    writer.add_graph(model, dummy)    

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config.epochs)
    
    print("\n" + "=" * 50)
    print("Training")
    print("=" * 50)
    
    best_accuracy = 0.0

    for epoch in range(config.epochs):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        writer.add_scalar("Loss/Train", train_loss, epoch)
        
        val_loss, val_acc = validate(model, test_loader, loss_fn, device)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Accuracy", val_acc, epoch)

        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar("Current LR", current_lr, epoch)

        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch)

        scheduler.step()

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            save_checkpoint(model, optimizer, scheduler, epoch, best_accuracy, "checkpoints/best_model.pth")

        print(f"Epoch {epoch+1}/{config.epochs}")
        print(f"Current LR: {current_lr:.4f}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Acc: {val_acc*100:.2f}%")
        print()

        save_checkpoint(model, optimizer, scheduler, epoch, best_accuracy, "checkpoints/last_checkpoint.pth")

    print("\n" + "=" * 50)
    print("Training completed")
    print("=" * 50)
    writer.close()
    
if __name__ == "__main__":
    main()