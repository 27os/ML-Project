import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from torchvision import datasets


FINE_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea",
    "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake",
    "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
    "tank", "telephone", "television", "tiger", "tractor", "train", "trout",
    "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman",
    "worm",
]

FINE_TO_SUPER = [
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
    3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
    0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
    16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
    2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
    18, 1, 2, 15, 6, 0, 17, 8, 14, 13,
]

SUPERCLASS_NAMES = [
    "aquatic mammals", "fish", "flowers", "food containers", "fruit & veg",
    "household electrical", "household furniture", "insects", "large carnivores",
    "large man-made outdoor", "large natural outdoor", "large omnivores/herbivores",
    "medium mammals", "non-insect invertebrates", "people", "reptiles",
    "small mammals", "trees", "vehicles 1", "vehicles 2",
]

RANDOM_STATE = 42


def load_features(feature_dir, backbone, split):
    X = np.load(os.path.join(feature_dir, f"{backbone}_{split}_features.npy"))
    y = np.load(os.path.join(feature_dir, f"{backbone}_{split}_labels.npy"))
    return X, y


def train_svm(X_train, y_train):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = SGDClassifier(
        loss="hinge", alpha=1e-4, max_iter=2000,
        tol=1e-3, random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(X_scaled, y_train)
    return model, scaler


def get_predictions(model, scaler, X_test):
    return model.predict(scaler.transform(X_test))


def load_raw_test_images(data_dir="./data"):
    dataset = datasets.CIFAR100(
        root=data_dir, train=False, download=True,
        transform=None,
    )
    images = np.stack([np.array(img) for img, _ in dataset])  # (10000, 32, 32, 3) uint8
    return images


def fname(label):
    return FINE_CLASSES[label].replace("_", " ")


def sname(label):
    return SUPERCLASS_NAMES[FINE_TO_SUPER[label]]


def same_superclass(a, b):
    return FINE_TO_SUPER[a] == FINE_TO_SUPER[b]


def pick_examples(true, pred18, pred50, n=4, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)

    catA = np.where((pred18 != true) & (pred50 == true))[0]
    catB = np.where(
        (pred18 != true) & (pred50 != true) &
        np.array([same_superclass(true[i], pred50[i]) for i in range(len(true))])
    )[0]
    catC = np.setdiff1d(np.where(pred50 != true)[0], catB)

    def sample(arr, k):
        k = min(k, len(arr))
        return rng.choice(arr, size=k, replace=False)

    idxA = sample(catA, n)
    idxB = sample(catB, n)
    idxC = sample(catC, n)
    return idxA, idxB, idxC


def plot_grid(images, true, pred18, pred50, idxA, idxB, idxC, save_path):
    fig, axes = plt.subplots(3, 4, figsize=(13, 10))
    fig.subplots_adjust(hspace=0.55, wspace=0.25)

    categories = [
        (idxA, "R18 wrong → R50 right", "#2ca02c"),   # green
        (idxB, "Both wrong (same superclass)", "#ff7f0e"),  # orange
        (idxC, "R50 wrong", "#d62728"),                     # red
    ]

    for row, (indices, row_title, color) in enumerate(categories):
        for col, idx in enumerate(indices):
            ax = axes[row, col]
            ax.imshow(images[idx])
            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(4)
                spine.set_visible(True)

            true_lbl   = fname(true[idx])
            pred18_lbl = fname(pred18[idx])
            pred50_lbl = fname(pred50[idx])
            super_lbl  = sname(true[idx])

            title = (
                f"True: {true_lbl}\n"
                f"({super_lbl})\n"
                f"R18: {pred18_lbl}\n"
                f"R50: {pred50_lbl}"
            )
            ax.set_title(title, fontsize=7.5, ha="center",
                         color="black", linespacing=1.4)

        axes[row, 0].set_ylabel(row_title, fontsize=9, fontweight="bold",
                                color=color, rotation=90, labelpad=8)

    fig.suptitle(
        "Error Analysis: ResNet-18 vs ResNet-50 SVM Predictions on CIFAR-100 Test Set",
        fontsize=12, fontweight="bold", y=1.01,
    )

    legend_patches = [
        mpatches.Patch(color="#2ca02c", label="R18 wrong, R50 right"),
        mpatches.Patch(color="#ff7f0e", label="Both wrong (same superclass)"),
        mpatches.Patch(color="#d62728", label="R50 wrong"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


def print_analysis(true, pred18, pred50):
    n = len(true)
    acc18 = (pred18 == true).mean()
    acc50 = (pred50 == true).mean()

    wrong18 = pred18 != true
    wrong50 = pred50 != true

    both_wrong = wrong18 & wrong50
    r18_only_wrong = wrong18 & ~wrong50
    r50_only_wrong = ~wrong18 & wrong50

    r50_err_idx = np.where(wrong50)[0]
    within_super = sum(
        same_superclass(true[i], pred50[i]) for i in r50_err_idx
    )

    print(f"\n{'='*55}")
    print("Error Analysis Summary")
    print(f"{'='*55}")
    print(f"Test set size          : {n}")
    print(f"R18 SVM accuracy       : {acc18:.4f}")
    print(f"R50 SVM accuracy       : {acc50:.4f}")
    print(f"R18 wrong, R50 right   : {r18_only_wrong.sum()}  "
          f"({100*r18_only_wrong.mean():.1f}% of test set)")
    print(f"R50 wrong, R18 right   : {r50_only_wrong.sum()}  "
          f"({100*r50_only_wrong.mean():.1f}%)")
    print(f"Both wrong             : {both_wrong.sum()}  "
          f"({100*both_wrong.mean():.1f}%)")
    print(f"R50 errors within same superclass: "
          f"{within_super}/{len(r50_err_idx)} "
          f"({100*within_super/max(len(r50_err_idx),1):.1f}%)")


def main():
    feature_dir = "./features"
    data_dir    = "./data"
    out_dir     = "./figures"
    os.makedirs(out_dir, exist_ok=True)

    print("Loading features...")
    X_train18, y_train = load_features(feature_dir, "resnet18", "train")
    X_test18,  y_test  = load_features(feature_dir, "resnet18", "test")
    X_train50, _       = load_features(feature_dir, "resnet50", "train")
    X_test50,  _       = load_features(feature_dir, "resnet50", "test")

    print("Training R18 SVM...")
    model18, scaler18 = train_svm(X_train18, y_train)
    print("Training R50 SVM...")
    model50, scaler50 = train_svm(X_train50, y_train)

    pred18 = get_predictions(model18, scaler18, X_test18)
    pred50 = get_predictions(model50, scaler50, X_test50)

    print_analysis(y_test, pred18, pred50)

    print("\nLoading raw test images...")
    images = load_raw_test_images(data_dir)

    idxA, idxB, idxC = pick_examples(y_test, pred18, pred50, n=4)

    save_path = os.path.join(out_dir, "error_analysis.png")
    plot_grid(images, y_test, pred18, pred50, idxA, idxB, idxC, save_path)
    print("Done.")


if __name__ == "__main__":
    main()
