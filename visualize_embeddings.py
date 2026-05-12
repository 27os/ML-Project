import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


SUPERCLASS_NAMES = [
    "aquatic mammals", "fish", "flowers", "food containers", "fruit & vegetables",
    "household electrical", "household furniture", "insects", "large carnivores",
    "large man-made outdoor", "large natural outdoor", "large omnivores & herbivores",
    "medium mammals", "non-insect invertebrates", "people", "reptiles",
    "small mammals", "trees", "vehicles 1", "vehicles 2",
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

N_SUPERCLASSES = 20
SAMPLE_SIZE = 2000
RANDOM_STATE = 42


def load_test_features(feature_dir: str, backbone: str):
    X = np.load(os.path.join(feature_dir, f"{backbone}_test_features.npy"))
    y = np.load(os.path.join(feature_dir, f"{backbone}_test_labels.npy"))
    return X, y


def subsample(X, y, n=SAMPLE_SIZE, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=n, replace=False)
    return X[idx], y[idx]


def to_superclass(y_fine):
    return np.array([FINE_TO_SUPER[c] for c in y_fine])


def make_colormap(n=N_SUPERCLASSES):
    cmap = plt.get_cmap("tab20", n)
    return [cmap(i) for i in range(n)]


def run_pca(X, n_components=2):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    return pca.fit_transform(X_scaled), pca


def run_tsne(X_pca, perplexity=40):
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return tsne.fit_transform(X_pca)


def scatter(ax, coords, superclasses, colors, title, alpha=0.5, s=8):
    for sc in range(N_SUPERCLASSES):
        mask = superclasses == sc
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[colors[sc]], s=s, alpha=alpha, linewidths=0, rasterized=True,
        )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


def add_legend(fig, colors):
    patches = [
        mpatches.Patch(color=colors[i], label=SUPERCLASS_NAMES[i])
        for i in range(N_SUPERCLASSES)
    ]
    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=5,
        fontsize=7.5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )


def plot_pca(coords18, coords50, superclasses18, superclasses50, colors, pca18, pca50, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("PCA of ResNet-18 vs ResNet-50 Embeddings (CIFAR-100 test set, 2000 samples)",
                 fontsize=13, y=1.01)

    var18 = pca18.explained_variance_ratio_
    var50 = pca50.explained_variance_ratio_

    scatter(axes[0], coords18, superclasses18, colors,
            f"ResNet-18  (PC1={var18[0]:.1%}, PC2={var18[1]:.1%})")
    scatter(axes[1], coords50, superclasses50, colors,
            f"ResNet-50  (PC1={var50[0]:.1%}, PC2={var50[1]:.1%})")

    add_legend(fig, colors)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


def plot_tsne(coords18, coords50, superclasses18, superclasses50, colors, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("t-SNE of ResNet-18 vs ResNet-50 Embeddings (CIFAR-100 test set, 2000 samples)",
                 fontsize=13, y=1.01)

    scatter(axes[0], coords18, superclasses18, colors, "ResNet-18")
    scatter(axes[1], coords50, superclasses50, colors, "ResNet-50")

    add_legend(fig, colors)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


def main():
    feature_dir = "./features"
    out_dir = "./figures"
    os.makedirs(out_dir, exist_ok=True)

    colors = make_colormap()

    print("Loading features...")
    X18, y18_fine = load_test_features(feature_dir, "resnet18")
    X50, y50_fine = load_test_features(feature_dir, "resnet50")

    print(f"Subsampling {SAMPLE_SIZE} points...")
    X18, y18_fine = subsample(X18, y18_fine)
    X50, y50_fine = subsample(X50, y50_fine)

    super18 = to_superclass(y18_fine)
    super50 = to_superclass(y50_fine)

    print("Running PCA...")
    pca_coords18, pca18 = run_pca(X18)
    pca_coords50, pca50 = run_pca(X50)
    plot_pca(pca_coords18, pca_coords50, super18, super50, colors, pca18, pca50,
             os.path.join(out_dir, "pca_comparison.png"))

    print("Running PCA(50) for t-SNE init...")
    _, pca50d_18 = run_pca(X18, n_components=50)
    X18_pca50 = StandardScaler().fit_transform(X18)
    X18_pca50 = PCA(n_components=50, random_state=RANDOM_STATE).fit_transform(X18_pca50)

    X50_pca50 = StandardScaler().fit_transform(X50)
    X50_pca50 = PCA(n_components=50, random_state=RANDOM_STATE).fit_transform(X50_pca50)

    print("Running t-SNE on ResNet-18 embeddings...")
    tsne_coords18 = run_tsne(X18_pca50)
    print("Running t-SNE on ResNet-50 embeddings...")
    tsne_coords50 = run_tsne(X50_pca50)

    plot_tsne(tsne_coords18, tsne_coords50, super18, super50, colors,
              os.path.join(out_dir, "tsne_comparison.png"))

    print("Done. Figures saved to ./figures/")


if __name__ == "__main__":
    main()