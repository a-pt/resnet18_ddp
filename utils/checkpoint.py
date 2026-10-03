import torch


def save_checkpoint(model, optimizer, scheduler, epoch, best_accuracy, path):
    
    model = model.module if hasattr(model, "module") else model

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_accuracy": best_accuracy,
    }

    torch.save(checkpoint, path)

def load_checkpoint(model, optimizer, scheduler, path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    
    model = model.module if hasattr(model, "module") else model

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint["epoch"]
    best_accuracy = checkpoint["best_accuracy"]

    return epoch, best_accuracy