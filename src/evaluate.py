"""
Evaluation utilities for the Deep Fingerprinting model.

Implements the metrics described in the paper's Appendix B:
  - Closed-world: accuracy, top-N accuracy
  - Open-world: TP, FP, TN, FN, TPR, FPR, Precision, Recall
    across varying prediction probability thresholds

Also provides plotting functions for precision-recall and ROC curves.
"""

import csv
import numpy as np


# ---------------------------------------------------------------------------
# Closed-World Metrics
# ---------------------------------------------------------------------------
def compute_top_n_accuracy(y_true, y_pred_probs, n=2):
    """Compute top-N accuracy.

    For Walkie-Talkie evaluation, the paper shows that top-2 accuracy
    reaches 98.44%, indicating the model correctly identifies both the
    real site and its decoy.

    Args:
        y_true: Array of true class indices, shape (num_samples,).
        y_pred_probs: Prediction probability matrix, shape (num_samples, num_classes).
        n: Number of top predictions to consider.

    Returns:
        Top-N accuracy as a float.
    """
    top_n_preds = np.argsort(y_pred_probs, axis=1)[:, -n:]
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] in top_n_preds[i]:
            correct += 1
    return correct / len(y_true)


# ---------------------------------------------------------------------------
# Open-World Metrics
# ---------------------------------------------------------------------------
def evaluate_open_world(pred_Mon, y_test_Mon, pred_Unmon, y_test_Unmon,
                        num_monitored, thresholds=None):
    """Evaluate open-world performance across thresholds.

    In the open-world scenario (paper Section 5.7), classification is done by:
      1. Getting the model's prediction probabilities for each trace.
      2. If the max probability for any MONITORED class exceeds a threshold,
         the trace is classified as monitored.
      3. Otherwise, it is classified as unmonitored.

    Args:
        pred_Mon: Prediction probabilities for monitored test traces,
                  shape (n_mon, num_classes).
        y_test_Mon: True labels for monitored test traces, shape (n_mon,).
        pred_Unmon: Prediction probabilities for unmonitored test traces,
                    shape (n_unmon, num_classes).
        y_test_Unmon: True labels for unmonitored test traces, shape (n_unmon,).
        num_monitored: Number of monitored site classes.
        thresholds: List of thresholds to evaluate. If None, uses
                    np.arange(0.0, 1.01, 0.01).

    Returns:
        List of dicts, each containing: threshold, TP, FP, TN, FN,
        TPR, FPR, Precision, Recall.
    """
    if thresholds is None:
        thresholds = np.arange(0.0, 1.01, 0.01)

    results = []

    for threshold in thresholds:
        TP = 0  # Monitored correctly classified as monitored
        FN = 0  # Monitored misclassified as unmonitored
        FP = 0  # Unmonitored misclassified as monitored
        TN = 0  # Unmonitored correctly classified as unmonitored

        # Evaluate monitored traces
        for i in range(len(pred_Mon)):
            # Max probability among monitored classes only
            mon_probs = pred_Mon[i, :num_monitored]
            max_prob = np.max(mon_probs)

            if max_prob >= threshold:
                TP += 1
            else:
                FN += 1

        # Evaluate unmonitored traces
        for i in range(len(pred_Unmon)):
            mon_probs = pred_Unmon[i, :num_monitored]
            max_prob = np.max(mon_probs)

            if max_prob >= threshold:
                FP += 1
            else:
                TN += 1

        # Compute rates
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        Precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        Recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

        results.append({
            'threshold': round(threshold, 4),
            'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
            'TPR': round(TPR, 6),
            'FPR': round(FPR, 6),
            'Precision': round(Precision, 6),
            'Recall': round(Recall, 6),
        })

    return results


def save_open_world_results(results, filepath):
    """Save open-world evaluation results to a CSV file.

    Args:
        results: List of dicts from evaluate_open_world().
        filepath: Path to the output CSV file.
    """
    fieldnames = ['threshold', 'TP', 'FP', 'TN', 'FN',
                  'TPR', 'FPR', 'Precision', 'Recall']
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Open-world results saved to {filepath}")


# ---------------------------------------------------------------------------
# Plotting Utilities
# ---------------------------------------------------------------------------
def plot_precision_recall(results, title='Precision-Recall Curve',
                          save_path=None):
    """Plot precision-recall curve from open-world results.

    Args:
        results: List of dicts from evaluate_open_world().
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    import matplotlib.pyplot as plt

    recalls = [r['Recall'] for r in results]
    precisions = [r['Precision'] for r in results]

    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, 'b-', linewidth=2, label='DF')

    # Baseline (random guessing)
    plt.axhline(y=0.5, color='gray', linestyle='--', label='Baseline')

    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Precision-Recall plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_roc_curve(results, title='ROC Curve', save_path=None):
    """Plot ROC curve from open-world results.

    Args:
        results: List of dicts from evaluate_open_world().
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    import matplotlib.pyplot as plt

    fprs = [r['FPR'] for r in results]
    tprs = [r['TPR'] for r in results]

    plt.figure(figsize=(8, 6))
    plt.plot(fprs, tprs, 'b-', linewidth=2, label='DF')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Baseline')

    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 0.35])
    plt.ylim([0, 1.05])
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"ROC plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_training_history(history, title='Training History', save_path=None):
    """Plot training/validation accuracy and loss curves.

    Reproduces Figure 4 from the paper showing training convergence.

    Args:
        history: Keras History object from model.fit().
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history.history['accuracy']) + 1)

    # Accuracy
    ax1.plot(epochs, history.history['accuracy'], 'b-', label='Training')
    ax1.plot(epochs, history.history['val_accuracy'], 'r-', label='Validation')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Accuracy', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Loss / Error Rate
    train_err = [1 - a for a in history.history['accuracy']]
    val_err = [1 - a for a in history.history['val_accuracy']]
    ax2.plot(epochs, train_err, 'b-', label='Training Error')
    ax2.plot(epochs, val_err, 'r-', label='Validation Error')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Error Rate', fontsize=12)
    ax2.set_title('Error Rate', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Training history plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_confusion_matrix(cm, class_names=None, title='Confusion Matrix',
                          save_path=None):
    """Plot row-normalized confusion matrix as a heatmap.

    Args:
        cm: Confusion matrix from compute_confusion_matrix(), shape (C, C).
        class_names: Optional list of class label strings.
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    import matplotlib.pyplot as plt

    n = cm.shape[0]
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)

    if class_names is not None:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(class_names, rotation=90, fontsize=6)
        ax.set_yticklabels(class_names, fontsize=6)

    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_per_class_accuracy(y_true, y_pred, class_names=None,
                            title='Per-Class Accuracy', save_path=None):
    """Bar chart of per-class accuracy sorted ascending.

    Args:
        y_true: Array of true class indices.
        y_pred: Array of predicted class indices.
        class_names: Optional list of class label strings.
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    num_classes = int(y_true.max()) + 1

    accs = np.array([(y_pred[y_true == c] == c).mean()
                     for c in range(num_classes)])
    order = np.argsort(accs)
    labels = ([class_names[i] for i in order] if class_names is not None
              else [str(i) for i in order])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(num_classes), accs[order], color='steelblue', width=0.8)
    ax.axhline(accs.mean(), color='red', linestyle='--', linewidth=1,
               label=f'Mean ({accs.mean():.3f})')
    ax.set_xticks(range(num_classes))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_xlabel('Website (sorted by accuracy)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Per-class accuracy plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_confidence_distribution(y_true, y_pred_probs,
                                 title='Prediction Confidence', save_path=None):
    """Histogram of max predicted probability for correct vs. incorrect samples.

    Args:
        y_true: Array of true class indices.
        y_pred_probs: Prediction probability matrix, shape (N, num_classes).
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    import matplotlib.pyplot as plt

    y_pred = np.argmax(y_pred_probs, axis=1)
    max_probs = np.max(y_pred_probs, axis=1)
    correct = max_probs[y_pred == y_true]
    incorrect = max_probs[y_pred != y_true]

    bins = np.linspace(0, 1, 40)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(correct, bins=bins, alpha=0.65, color='steelblue',
            label=f'Correct (n={len(correct)})')
    ax.hist(incorrect, bins=bins, alpha=0.65, color='tomato',
            label=f'Incorrect (n={len(incorrect)})')
    ax.set_xlabel('Max predicted probability', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Confidence distribution saved to {save_path}")
    else:
        plt.show()
    plt.close()


def compute_confusion_matrix(y_true, y_pred, num_classes):
    """Compute confusion matrix.

    Args:
        y_true: True class indices.
        y_pred: Predicted class indices.
        num_classes: Number of classes.

    Returns:
        Confusion matrix of shape (num_classes, num_classes).
    """
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm
