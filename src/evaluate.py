"""
evaluate.py — Evaluation metrics, predictions, and analysis utilities.
"""

import torch
import numpy as np
import time
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, top_k_accuracy_score


@torch.no_grad()
def get_all_predictions(model, loader, device):
    """
    Run model on entire data loader.

    Returns:
        preds: (N,) predicted class indices
        labels: (N,) ground truth class indices
        probs: (N, C) softmax probabilities
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in tqdm(loader, desc='Evaluating'):
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def compute_accuracy(preds, labels, probs):
    """Compute Top-1 and Top-5 accuracy."""
    top1 = 100. * np.mean(preds == labels)
    top5 = 100. * top_k_accuracy_score(labels, probs, k=5)
    return top1, top5


def get_classification_report(preds, labels, class_names, output_dict=True):
    """Generate sklearn classification report."""
    names = [c.replace('_', ' ') for c in class_names]
    return classification_report(labels, preds, target_names=names,
                                 output_dict=output_dict)


def get_confusion_matrix(preds, labels):
    """Generate confusion matrix."""
    return confusion_matrix(labels, preds)


def get_most_confused_pairs(cm, class_names, top_k=20):
    """
    Find the most confused class pairs from a confusion matrix.

    Returns:
        List of (true_class, predicted_class, count) tuples, sorted descending.
    """
    pairs = []
    n = len(class_names)
    for i in range(n):
        for j in range(n):
            if i != j and cm[i][j] > 0:
                pairs.append((
                    class_names[i].replace('_', ' '),
                    class_names[j].replace('_', ' '),
                    int(cm[i][j])
                ))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_k]


def measure_inference_speed(model, device, num_runs=50):
    """Measure average inference time per image in milliseconds."""
    model.eval()
    dummy = torch.randn(1, 3, 224, 224).to(device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(dummy)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    elapsed = time.time() - start
    return (elapsed / num_runs) * 1000
