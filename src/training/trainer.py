from pathlib import Path
from typing import Dict, List, Tuple
import copy
import time

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm


def calculate_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(outputs, dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct / total


def run_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: Optimizer | None,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    is_train: bool,
) -> Tuple[float, float]:
    if is_train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_acc = 0.0
    total_batches = len(dataloader)

    pbar = tqdm(dataloader, desc="Train" if is_train else "Val", leave=False)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            use_amp = device.type == "cuda"

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            acc = calculate_accuracy(outputs, labels)

            if is_train:
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        running_loss += loss.item()
        running_acc += acc

        pbar.set_postfix(
            loss=f"{running_loss / (pbar.n / images.size(0) + 1e-8):.4f}",
            acc=f"{running_acc / (pbar.n / images.size(0) + 1e-8):.4f}",
        )

    epoch_loss = running_loss / total_batches
    epoch_acc = running_acc / total_batches
    return epoch_loss, epoch_acc


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: _LRScheduler | None,
    device: torch.device,
    num_epochs: int,
    checkpoint_path: str | Path,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_time_sec": [],
    }

    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
        start_time = time.time()

        train_loss, train_acc = run_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            is_train=True,
        )

        val_loss, val_acc = run_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            scaler=None,
            is_train=False,
        )

        if scheduler is not None:
            scheduler.step()

        epoch_time = time.time() - start_time

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_time_sec"].append(epoch_time)

        print(
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | "
            f"time={epoch_time:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_val_acc,
                    "history": history,
                },
                checkpoint_path,
            )
            print(f"Saved best checkpoint to: {checkpoint_path}")

    model.load_state_dict(best_model_wts)
    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    return model, history