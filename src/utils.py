import torch
import numpy as np
import torch.nn as nn
from torchvision import models
from torchvision.models.efficientnet import MBConv
from tqdm.notebook import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

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

def get_all_embeddings(model, loader, device):
    """Helper function to extract embeddings and labels for a full dataset."""
    model.eval()
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            embeddings = model(xb)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(yb.cpu().numpy())
    return np.vstack(all_embeddings), np.concatenate(all_labels)


def evaluate_knn(model, gallery_loader, test_loader, device, n_neighbors=5):
    """
    Evaluates the embedding model using a k-NN classifier, returning a full dict of metrics.
    """
    train_embeddings, train_labels = get_all_embeddings(model, gallery_loader, device)
    test_embeddings, test_labels = get_all_embeddings(model, test_loader, device)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
    knn.fit(train_embeddings, train_labels)

    y_pred_knn = knn.predict(test_embeddings)
    
    f1 = f1_score(test_labels, y_pred_knn, average="macro", zero_division=0)
    precision = precision_score(test_labels, y_pred_knn, average="macro", zero_division=0)
    recall = recall_score(test_labels, y_pred_knn, average="macro", zero_division=0)
    accuracy = accuracy_score(test_labels, y_pred_knn)
    
    return {
        'f1_macro': f1,
        'precision_macro': precision,
        'recall_macro': recall,
        'accuracy': accuracy
    }

def get_knn_predictions(model, gallery_loader, test_loader, device, n_neighbors=5):
    """
    Fits a k-NN on the gallery embeddings and returns true labels and predictions
    for the test set. Used for confusion matrix and classification reports.
    """
    train_embeddings, train_labels = get_all_embeddings(model, gallery_loader, device)
    test_embeddings, test_labels = get_all_embeddings(model, test_loader, device)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
    knn.fit(train_embeddings, train_labels)

    y_pred_knn = knn.predict(test_embeddings)

    return test_labels, y_pred_knn


def load_backbone_for_embedding(model_path, device):
    import torch, torch.nn as nn
    from torchvision import models

    model = models.efficientnet_b0()
    state = torch.load(model_path, map_location="cpu")
    state = {k: v for k, v in state.items() if not k.startswith("classifier.")}
    model.classifier = nn.Identity()
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    with torch.no_grad():
        z = model(torch.zeros(1,3,224,224, device=device))
        emb_dim = int((z[0] if isinstance(z,(tuple,list)) else z).shape[-1])

    print(f"Loaded embedding model with output dimension {emb_dim}.")
    return model