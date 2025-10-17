import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

def train_step(
    model, loader, criterion, optimizer, device,
    disable_tqdm=False,
    scheduler=None,
    grad_clip=None,
    accumulate_steps=1
):
    model.train()
    total_loss = 0.0
    all_preds_cpu = []
    all_true_cpu = []
    correct = 0

    optimizer.zero_grad(set_to_none=True)
    num_batches = 0

    for step, (xb, yb) in enumerate(tqdm(loader, desc="Training", leave=False, disable=disable_tqdm), start=1):
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = criterion(logits, yb) / accumulate_steps
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        if step % accumulate_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        bs = xb.size(0)
        total_loss += loss.item() * bs * accumulate_steps
        preds = logits.argmax(1)

        correct += (preds == yb).sum().item()

        all_preds_cpu.append(preds.detach().cpu())
        all_true_cpu.append(yb.detach().cpu())

        num_batches += 1

    if (num_batches % accumulate_steps) != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()

    avg_loss = total_loss / len(loader.dataset)

    y_pred = torch.cat(all_preds_cpu, dim=0).numpy()
    y_true = torch.cat(all_true_cpu, dim=0).numpy()

    accuracy = float((y_pred == y_true).mean())
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1_macro': f1,
        'precision_macro': precision,
        'recall_macro': recall
    }


def val_step(model, loader, criterion, device, disable_tqdm=False):
    model.eval()
    total_loss = 0.0
    all_preds_cpu = []
    all_true_cpu = []

    with torch.no_grad():
        for xb, yb in tqdm(loader, desc="Validation", leave=False, disable=disable_tqdm):
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = criterion(logits, yb)

            bs = xb.size(0)
            total_loss += loss.item() * bs

            preds = logits.argmax(1)
            all_preds_cpu.append(preds.detach().cpu())
            all_true_cpu.append(yb.detach().cpu())

    avg_loss = total_loss / len(loader.dataset)

    y_pred = torch.cat(all_preds_cpu, dim=0).numpy()
    y_true = torch.cat(all_true_cpu, dim=0).numpy()

    accuracy = float((y_pred == y_true).mean())
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1_macro': f1,
        'precision_macro': precision,
        'recall_macro': recall
    }


def train_step_metric(model, loader, criterion, optimizer, miner, device, scheduler=None, disable_tqdm=False):
    """Performs a single training epoch for a metric learning model."""
    model.train()
    total_loss = 0

    pbar = tqdm(loader, desc="Training", leave=False, disable=disable_tqdm)
    for xb, yb in pbar:
        xb, yb = xb.to(device), yb.to(device)
        
        optimizer.zero_grad()
        
        embeddings = model(xb)
        hard_triplets = miner(embeddings, yb)
        loss = criterion(embeddings, yb, hard_triplets)
        
        if loss is not None: # Miner can sometimes return no triplets
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    if scheduler:
        scheduler.step()
        
    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
    return {'loss': avg_loss}