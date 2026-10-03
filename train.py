from torch import device
import time

import torch
import torch.nn as nn

from torch.profiler import profile, record_function, ProfilerActivity

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.checkpoint import save_checkpoint
from configs.config import Config
from utils.device import get_device
from utils.seed import set_seed
from datasets.cifar10 import get_cifar10_datasets
from models.resnet18 import ResNet18

from engine import train_one_epoch, validate

def profile_training(model, train_loader, loss_fn, optimizer, scaler, use_amp, device):
    
    model.train()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./runs/resnet18_profiler"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:

        for step, (images, labels) in enumerate(train_loader):

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with record_function("forward"):
                with torch.autocast(
                    device_type=device.type,
                    enabled=use_amp,
                ):
                    logits = model(images)
                    loss = loss_fn(logits, labels)

            with record_function("backward"):
                scaler.scale(loss).backward()

            with record_function("optimizer_step"):
                scaler.step(optimizer)
                scaler.update()

            prof.step()

            if step >= 4:
                break
    
    print("\n" + "=" * 50)
    print("Profiler Results")
    print("=" * 50)
    print(prof.key_averages().table(sort_by="self_cuda_time_total",row_limit=20))

def main():
    config = Config()
    set_seed(config.seed)
    device = get_device()

    # cuDNN benchmark
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False

    # AMP configuration
    use_amp = device.type == "cuda"

    scaler = torch.amp.GradScaler("cuda",enabled=use_amp)

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

    # print("\nRunning PyTorch Profiler...")
    # profile_training(model, train_loader, loss_fn, optimizer, scaler, use_amp, device)
    # print("Profiler completed.")
    
    # return
    
    print("\n" + "=" * 50)
    print("Training")
    print("=" * 50)
    
    best_accuracy = 0.0

    for epoch in range(config.epochs):

        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, scaler, use_amp, device)
        
        if device.type == "cuda":
            torch.cuda.synchronize()

        train_time = time.perf_counter() - start_time

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
        print(f"Train Time: {train_time:.2f} seconds")
        print()

        save_checkpoint(model, optimizer, scheduler, epoch, best_accuracy, "checkpoints/last_checkpoint.pth")

    print("\n" + "=" * 50)
    print("Training completed")
    print("=" * 50)
    writer.close()
    
if __name__ == "__main__":
    main()