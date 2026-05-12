import os
import sys
import time
import itertools
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score


class Tee:
    def __init__(self, filename):
        self.file = open(filename, "w")
        self.stdout = sys.stdout

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def load_split(feature_dir: str, backbone: str, split: str):
    X = np.load(os.path.join(feature_dir, f"{backbone}_{split}_features.npy"))
    y = np.load(os.path.join(feature_dir, f"{backbone}_{split}_labels.npy"))
    return X, y


def maybe_standardize(X_train, X_val, X_test):
    scaler = StandardScaler()
    return (
        scaler.fit_transform(X_train),
        scaler.transform(X_val),
        scaler.transform(X_test),
    )


def train_and_eval_mlp(X_train, y_train, X_val, y_val, hidden, alpha, lr):
    model = MLPClassifier(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        alpha=alpha,
        batch_size=256,
        learning_rate_init=lr,
        max_iter=50,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=7,
        random_state=42,
        verbose=False,
    )
    model.fit(X_train, y_train)
    val_acc = accuracy_score(y_val, model.predict(X_val))
    return model, val_acc


def full_eval(model, X, y, split_name):
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average="macro")
    top5 = top_k_accuracy_score(y, model.predict_proba(X), k=5)
    print(f"  {split_name:<6} | Acc: {acc:.4f} | Top-5: {top5:.4f} | Macro-F1: {f1:.4f}")
    return acc, top5, f1


def tune_backbone(backbone: str, feature_dir: str):
    print(f"\n{'=' * 70}")
    print(f"Tuning MLP — Backbone: {backbone}")
    print(f"{'=' * 70}")

    X_train, y_train = load_split(feature_dir, backbone, "train")
    X_val,   y_val   = load_split(feature_dir, backbone, "val")
    X_test,  y_test  = load_split(feature_dir, backbone, "test")
    X_train, X_val, X_test = maybe_standardize(X_train, X_val, X_test)

    # Grid: hidden sizes, regularization strength, learning rate
    hidden_options = [(512,), (1024,), (512, 256), (1024, 512)]
    alpha_options  = [1e-5, 1e-4, 1e-3, 1e-2]
    lr_options     = [1e-4, 5e-4, 1e-3]

    grid = list(itertools.product(hidden_options, alpha_options, lr_options))
    print(f"Grid size: {len(grid)} configs\n")

    results = []
    for i, (hidden, alpha, lr) in enumerate(grid, 1):
        t0 = time.time()
        model, val_acc = train_and_eval_mlp(X_train, y_train, X_val, y_val, hidden, alpha, lr)
        elapsed = time.time() - t0
        tag = f"hidden={hidden} alpha={alpha:.0e} lr={lr:.0e}"
        print(f"[{i:>3}/{len(grid)}] {tag:<45} val_acc={val_acc:.4f}  ({elapsed:.1f}s)")
        results.append((val_acc, model, hidden, alpha, lr))

    results.sort(key=lambda x: x[0], reverse=True)

    print(f"\n--- Top-5 configs by val accuracy ---")
    for rank, (val_acc, _, hidden, alpha, lr) in enumerate(results[:5], 1):
        print(f"  #{rank}: hidden={hidden} alpha={alpha:.0e} lr={lr:.0e}  val_acc={val_acc:.4f}")

    best_val_acc, best_model, best_hidden, best_alpha, best_lr = results[0]

    print(f"\n--- Best config: hidden={best_hidden} alpha={best_alpha:.0e} lr={best_lr:.0e} ---")
    print(f"Full evaluation on best model:")
    train_acc, train_top5, train_f1 = full_eval(best_model, X_train, y_train, "train")
    val_acc,   val_top5,   val_f1   = full_eval(best_model, X_val,   y_val,   "val")
    test_acc,  test_top5,  test_f1  = full_eval(best_model, X_test,  y_test,  "test")

    return {
        "backbone":    backbone,
        "hidden":      best_hidden,
        "alpha":       best_alpha,
        "lr":          best_lr,
        "val_acc":     val_acc,
        "val_top5":    val_top5,
        "val_f1":      val_f1,
        "test_acc":    test_acc,
        "test_top5":   test_top5,
        "test_f1":     test_f1,
    }


def print_final_summary(all_results):
    print(f"\n{'=' * 95}")
    print("TUNING FINAL SUMMARY — Best MLP per backbone")
    print(f"{'=' * 95}")
    print(
        f"{'Backbone':<12} {'Hidden':<16} {'Alpha':<8} {'LR':<8} "
        f"{'Val Acc':<10} {'Val Top-5':<10} "
        f"{'Test Acc':<10} {'Test Top-5':<11} {'Test F1':<10}"
    )
    print("-" * 95)
    for r in all_results:
        print(
            f"{r['backbone']:<12} {str(r['hidden']):<16} {r['alpha']:<8.0e} {r['lr']:<8.0e} "
            f"{r['val_acc']:<10.4f} {r['val_top5']:<10.4f} "
            f"{r['test_acc']:<10.4f} {r['test_top5']:<11.4f} {r['test_f1']:<10.4f}"
        )


def main():
    sys.stdout = Tee("tune_mlp_log.txt")
    feature_dir = "./features"
    backbones = ["resnet18", "resnet50"]

    all_results = []
    for backbone in backbones:
        result = tune_backbone(backbone, feature_dir)
        all_results.append(result)

    print_final_summary(all_results)


if __name__ == "__main__":
    main()