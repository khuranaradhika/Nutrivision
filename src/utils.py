import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from .dataset import denormalize


def plot_training_curves(history, model_name, save_path=None):
    epochs = range(1, len(history['train_loss']) + 1)
    phase_change = sum(1 for p in history['phase'] if p == 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history['train_loss'], 'b-', label='Train')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val')
    ax1.axvline(x=phase_change + 0.5, color='gray', linestyle='--', alpha=0.7, label='Phase 2')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name} — Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history['train_acc'], 'b-', label='Train')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val')
    ax2.axvline(x=phase_change + 0.5, color='gray', linestyle='--', alpha=0.7, label='Phase 2')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{model_name} — Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(cm, class_names, save_path=None, figsize=(24, 20)):
    fig, ax = plt.subplots(figsize=figsize)
    names = [c.replace('_', ' ') for c in class_names]
    sns.heatmap(cm, cmap='Blues', ax=ax,
                xticklabels=names, yticklabels=names,
                cbar_kws={'shrink': 0.6})
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14)
    ax.tick_params(axis='both', labelsize=5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()


def plot_worst_classes_cm(cm, class_names, sorted_classes, top_k=15, save_path=None):
    worst_indices = [class_names.index(n.replace(' ', '_')) for n, _ in sorted_classes[-top_k:]]
    cm_worst = cm[np.ix_(worst_indices, worst_indices)]
    worst_names = [class_names[i].replace('_', ' ') for i in worst_indices]

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_worst, annot=True, fmt='d', cmap='Reds',
                xticklabels=worst_names, yticklabels=worst_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix — {top_k} Hardest Classes')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def show_sample_predictions(dataset, preds, labels, class_names, correct=True, n=10, save_path=None):
    mask = (preds == labels) if correct else (preds != labels)
    indices = np.where(mask)[0]
    np.random.seed(42)
    sample = np.random.choice(indices, min(n, len(indices)), replace=False)

    rows = (len(sample) + 4) // 5
    fig, axes = plt.subplots(rows, 5, figsize=(18, 4 * rows))
    fig.suptitle(f'Sample {"Correct" if correct else "Incorrect"} Predictions', fontsize=14)

    for ax, idx in zip(axes.flat, sample):
        img, _ = dataset[idx]
        img_display = denormalize(img).permute(1, 2, 0).numpy()
        true_name = class_names[labels[idx]].replace('_', ' ').title()
        pred_name = class_names[preds[idx]].replace('_', ' ').title()

        ax.imshow(img_display)
        if correct:
            ax.set_title(f'✓ {true_name}', fontsize=9, color='green')
        else:
            ax.set_title(f'True: {true_name}\nPred: {pred_name}', fontsize=9, color='red')
        ax.axis('off')

    for ax in axes.flat[len(sample):]:
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def show_top5_predictions(model, dataset, indices, class_names, device, save_path=None):
    model.eval()
    n = len(indices)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n), gridspec_kw={'width_ratios': [1, 2]})
    if n == 1:
        axes = axes.reshape(1, -1)

    for row, idx in enumerate(indices):
        img, label = dataset[idx]
        img_display = denormalize(img).permute(1, 2, 0).numpy()

        with torch.no_grad():
            output = model(img.unsqueeze(0).to(device))
            probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()

        top5_idx = probs.argsort()[-5:][::-1]
        top5_probs = probs[top5_idx]
        top5_names = [class_names[i].replace('_', ' ').title() for i in top5_idx]
        true_name = class_names[label].replace('_', ' ').title()

        axes[row, 0].imshow(img_display)
        axes[row, 0].set_title(f'True: {true_name}', fontsize=11)
        axes[row, 0].axis('off')

        colors = ['green' if n == true_name else 'salmon' for n in top5_names]
        bars = axes[row, 1].barh(range(5), top5_probs, color=colors)
        axes[row, 1].set_yticks(range(5))
        axes[row, 1].set_yticklabels(top5_names)
        axes[row, 1].set_xlim(0, 1)
        axes[row, 1].set_xlabel('Confidence')
        axes[row, 1].invert_yaxis()

        for bar, prob in zip(bars, top5_probs):
            axes[row, 1].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                              f'{prob:.1%}', va='center', fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
