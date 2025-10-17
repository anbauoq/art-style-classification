import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.efficientnet import MBConv
from tqdm.notebook import tqdm

def set_device():
    """Sets the device to CUDA, MPS, or CPU."""
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}")
    return device

def is_mbconv(m):
    """Checks if a module is an MBConv block."""
    return (MBConv is not None and isinstance(m, MBConv)) or (m.__class__.__name__ == "MBConv")

class EarlyStopper:
    """
    Stops training based on two conditions:
      1. A monitored validation metric doesn't improve for `patience` epochs.
      2. The overfitting gap (val_metric - train_metric) exceeds `max_overfit_gap`.
    """
    def __init__(self, mode="max", patience=5, max_overfit_gap=0.10):
        assert mode in ("min", "max"), "Mode must be 'min' or 'max'."
        self.mode = mode
        self.patience = patience
        self.max_overfit_gap = max_overfit_gap
        self.best_metric = float('-inf') if mode == "max" else float('inf')
        self.epochs_no_improve = 0
        self.reason = None

    def step(self, val_patience_metric, train_metric=None, val_metric=None):
        improved = (self.mode == "max" and val_patience_metric > self.best_metric) or \
                   (self.mode == "min" and val_patience_metric < self.best_metric)

        if improved:
            self.best_metric = val_patience_metric
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1

        stop_due_to_patience = self.epochs_no_improve >= self.patience
        
        stop_due_to_gap = False
        gap = 0
        if self.max_overfit_gap is not None and train_metric is not None and val_metric is not None:
            gap = val_metric - train_metric
            if gap > self.max_overfit_gap:
                stop_due_to_gap = True

        should_stop = stop_due_to_patience or stop_due_to_gap
        if should_stop:
            reasons = []
            if stop_due_to_patience:
                reasons.append(f"patience of {self.patience} epochs was reached")
            if stop_due_to_gap:
                reasons.append(f"overfitting gap of {gap:.4f} exceeded the max of {self.max_overfit_gap}")
            self.reason = " and ".join(reasons)

        return improved, should_stop, self.reason

def get_predictions(model, loader, device, show_progress=False):
    """Runs inference and returns predictions and true labels."""
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in tqdm(loader, disable=not show_progress):
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds = logits.argmax(1)
            all_preds.append(preds.cpu())
            all_true.append(yb.cpu())
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_true).numpy()
    return y_true, y_pred