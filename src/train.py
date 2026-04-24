import torch
import torch.nn as nn
import torch.optim as optim
import copy
from tqdm import tqdm


def train_one_epoch(model, loader, criterion, optimizer, device, epoch=1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f'Epoch {epoch}', leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100. * correct / total:.1f}%'})

    return running_loss / total, 100. * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100. * correct / total


def train_two_phase(model, model_name, train_loader, val_loader, device,
                    phase1_epochs=5, phase2_epochs=15,
                    head_lr=1e-3, finetune_lr_head=1e-4,
                    finetune_lr_backbone=1e-5, patience=5):
    """
    Phase 1: train only the classification head (backbone frozen).
    Phase 2: fine-tune the whole network with differential LRs.
    """
    from .models import unfreeze_model, get_param_groups

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'phase': []}
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    best_weights = None

    print(f'\nPhase 1 — head only ({phase1_epochs} epochs)')
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=head_lr)

    for epoch in range(1, phase1_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['phase'].append(1)

        print(f'  [{epoch}/{phase1_epochs}] train {train_acc:.2f}% | val {val_acc:.2f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())

    print(f'\nPhase 2 — full fine-tune ({phase2_epochs} epochs)')
    unfreeze_model(model)
    param_groups = get_param_groups(model, model_name,
                                    backbone_lr=finetune_lr_backbone, head_lr=finetune_lr_head)
    optimizer = optim.Adam(param_groups)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    epochs_no_improve = 0

    for epoch in range(1, phase2_epochs + 1):
        global_epoch = epoch + phase1_epochs
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, global_epoch)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['phase'].append(2)

        scheduler.step(val_loss)
        print(f'  [{global_epoch}/{phase1_epochs + phase2_epochs}] train {train_acc:.2f}% | val {val_acc:.2f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f'Early stopping at epoch {global_epoch}')
            break

    model.load_state_dict(best_weights)
    print(f'\nBest val acc: {best_val_acc:.2f}%')
    return model, history
