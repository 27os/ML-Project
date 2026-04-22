import os
import time
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
import sys

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
    x_path = os.path.join(feature_dir, f"{backbone}_{split}_features.npy")
    y_path = os.path.join(feature_dir, f"{backbone}_{split}_labels.npy")

    X = np.load(x_path)
    y = np.load(y_path)

    print(f"Loaded {x_path}, shape={X.shape}")
    print(f"Loaded {y_path}, shape={y.shape}")
    return X, y


def evaluate_classifier(model, X, y, split_name="test"):
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    macro_f1 = f1_score(y, preds, average="macro")
    print(f"{split_name} | Acc: {acc:.4f} | Macro-F1: {macro_f1:.4f}")
    return acc, macro_f1

def train_fast_svm(X_train, y_train):
    model = SGDClassifier(
        loss="hinge",          # linear SVM-style objective
        alpha=1e-4,
        max_iter=2000,
        tol=1e-3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model

def maybe_standardize(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled


def train_logistic_regression(X_train, y_train):
    # removed deprecated multi_class argument
    model = LogisticRegression(
        max_iter=300,
        solver="lbfgs",
        n_jobs=-1,
        verbose=0,
    )
    model.fit(X_train, y_train)
    return model


def train_mlp(X_train, y_train):
    model = MLPClassifier(
        hidden_layer_sizes=(512,),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=30,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=42,
        verbose=False,
    )
    model.fit(X_train, y_train)
    return model


def run_one_experiment(backbone: str, classifier_name: str, feature_dir: str = "./features"):
    print("\n" + "=" * 70)
    print(f"Backbone: {backbone} | Classifier: {classifier_name}")
    print("=" * 70)

    X_train, y_train = load_split(feature_dir, backbone, "train")
    X_val, y_val = load_split(feature_dir, backbone, "val")
    X_test, y_test = load_split(feature_dir, backbone, "test")

    X_train, X_val, X_test = maybe_standardize(X_train, X_val, X_test)

    start_time = time.time()

    if classifier_name == "logreg":
        model = train_logistic_regression(X_train, y_train)
    elif classifier_name == "mlp":
        model = train_mlp(X_train, y_train)
    elif classifier_name == "svm_fast":
        model = train_fast_svm(X_train, y_train)
        
    else:
        raise ValueError("classifier_name must be one of: logreg, mlp")

    train_time = time.time() - start_time
    print(f"Training time: {train_time:.2f}s")

    train_acc, train_f1 = evaluate_classifier(model, X_train, y_train, split_name="train")
    val_acc, val_f1 = evaluate_classifier(model, X_val, y_val, split_name="val")
    test_acc, test_f1 = evaluate_classifier(model, X_test, y_test, split_name="test")

    result = {
        "backbone": backbone,
        "classifier": classifier_name,
        "train_acc": train_acc,
        "train_f1": train_f1,
        "val_acc": val_acc,
        "val_f1": val_f1,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "train_time_sec": train_time,
    }

    return result


def print_summary(results):
    print("\n" + "=" * 90)
    print("FINAL SUMMARY")
    print("=" * 90)
    print(
        f"{'Backbone':<12} {'Classifier':<12} "
        f"{'Val Acc':<10} {'Val F1':<10} "
        f"{'Test Acc':<10} {'Test F1':<10} "
        f"{'Train Time(s)':<12}"
    )
    print("-" * 90)

    for r in results:
        print(
            f"{r['backbone']:<12} {r['classifier']:<12} "
            f"{r['val_acc']:<10.4f} {r['val_f1']:<10.4f} "
            f"{r['test_acc']:<10.4f} {r['test_f1']:<10.4f} "
            f"{r['train_time_sec']:<12.2f}"
        )


def main():
    log_filename = "training_classifiers.txt"
    sys.stdout = Tee(log_filename)
    feature_dir = "./features"

    backbones = ["resnet18", "resnet50"]
    classifiers = ["logreg", "svm_fast", "mlp"]

    results = []

    for backbone in backbones:
        for clf_name in classifiers:
            result = run_one_experiment(
                backbone=backbone,
                classifier_name=clf_name,
                feature_dir=feature_dir,
            )
            results.append(result)

    print_summary(results)


if __name__ == "__main__":
    main()