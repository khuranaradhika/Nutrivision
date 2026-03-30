"""
models.py — Model factory for food classification architectures.
"""

import torch
import torch.nn as nn
import torchvision.models as models


def create_model(model_name='resnet50', num_classes=101, freeze_backbone=True,
                 device=None):
    """
    Create a model with pre-trained backbone and new classification head.

    Args:
        model_name: 'resnet50' or 'mobilenetv3'
        num_classes: number of output classes
        freeze_backbone: if True, freeze all layers except classification head
        device: torch device (auto-detected if None)

    Returns:
        model on the specified device
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_name == 'resnet50':
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )
        classifier_params = list(model.fc.parameters())

    elif model_name == 'mobilenetv3':
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
        model = models.mobilenet_v3_large(weights=weights)
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_features, num_classes)
        classifier_params = list(model.classifier.parameters())

    else:
        raise ValueError(f'Unknown model: {model_name}. Use "resnet50" or "mobilenetv3".')

    # Freeze backbone if requested
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in classifier_params:
            param.requires_grad = True

    model = model.to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{model_name.upper()} | Total: {total:,} | Trainable: {trainable:,}')

    return model


def unfreeze_model(model):
    """Unfreeze all model parameters for fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'All parameters unfrozen: {trainable:,} trainable')


def get_param_groups(model, model_name, backbone_lr=1e-5, head_lr=1e-4):
    """
    Create parameter groups with differential learning rates.

    Args:
        model: the model
        model_name: 'resnet50' or 'mobilenetv3'
        backbone_lr: learning rate for backbone layers
        head_lr: learning rate for classification head

    Returns:
        list of param group dicts for optimizer
    """
    if model_name == 'resnet50':
        backbone_params = [
            p for name, p in model.named_parameters() if 'fc' not in name
        ]
        head_params = list(model.fc.parameters())
    elif model_name == 'mobilenetv3':
        backbone_params = [
            p for name, p in model.named_parameters() if 'classifier' not in name
        ]
        head_params = list(model.classifier.parameters())
    else:
        raise ValueError(f'Unknown model: {model_name}')

    return [
        {'params': backbone_params, 'lr': backbone_lr},
        {'params': head_params, 'lr': head_lr}
    ]
