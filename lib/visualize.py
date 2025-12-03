import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

def plot_losses(losses):
    plt.figure(figsize=(6,4))
    plt.plot(losses)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

def plot_decision_boundary(model, X, y, resolution=300):
    """
    Automatically detects if the task is binary (y = [-1,1] or [0,1])
    or multi-class (one-hot vectors), then calls the appropriate plotter.
    """

    # Detect binary vs multiclass
    if y.ndim == 1 or y.shape[1] == 1:
        _plot_binary_boundary(model, X, y, resolution)
    else:
        _plot_multiclass_boundary(model, X, y, resolution)


# ----------------------------------------------------------------------
#       BINARY CLASSIFICATION PLOTTER
# ----------------------------------------------------------------------
def _plot_binary_boundary(model, X, y, resolution):
    print("\n" + "=" * 70)
    print("VISUALIZATION: Binary Decision Boundary")
    print("=" * 70)

    # Prepare grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )

    # Predict on grid
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.forward(grid).reshape(xx.shape)

    # Create figure
    fig, ax = plt.subplots(figsize=(11, 9))

    # Soft smooth contour
    contourf = ax.contourf(xx, yy, Z, levels=50, cmap="RdBu_r", alpha=0.8)
    ax.contour(xx, yy, Z, levels=[0], colors="black", linewidths=3)

    # Plot training points
    for i in range(len(X)):
        x_pt, y_pt = X[i]
        target = y[i][0]

        color = "red" if target > 0 else "blue"
        label_txt = "+1" if target > 0 else "-1"

        ax.scatter(
            x_pt, y_pt, 
            c=color, s=300, marker="o",
            edgecolors="black", linewidth=2.2, zorder=10
        )

        ax.text(
            x_pt, y_pt - 0.12,
            f"({int(x_pt)}, {int(y_pt)})",
            ha="center", fontsize=10, fontweight="bold"
        )

    # Labels & style
    ax.set_title(
        "Binary Decision Boundary (Your Neural Network)",
        fontsize=16, fontweight="bold"
    )
    ax.set_xlabel("x₁", fontsize=14)
    ax.set_ylabel("x₂", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Colorbar
    cbar = plt.colorbar(contourf, ax=ax)
    cbar.set_label("Network Output", fontsize=13)

    # Legend
    legend_elements = [
        Patch(facecolor="red", label="Class +1"),
        Patch(facecolor="blue", label="Class -1")
    ]
    ax.legend(handles=legend_elements, fontsize=12, loc="upper right")

    plt.tight_layout()
    plt.show()

    print("=" * 70)
    print("Binary decision boundary visualization complete!")
    print("=" * 70)


# ----------------------------------------------------------------------
#      MULTI-CLASS CLASSIFICATION PLOTTER
# ----------------------------------------------------------------------
def _plot_multiclass_boundary(model, X, y, resolution):
    print("\n" + "=" * 70)
    print("VISUALIZATION: Multi-Class Decision Boundary")
    print("=" * 70)

    # Grid range
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )

    # Predict on grid
    grid = np.c_[xx.ravel(), yy.ravel()]
    logits = model.forward(grid)
    Z = np.argmax(logits, axis=1).reshape(xx.shape)

    # Figure
    fig, ax = plt.subplots(figsize=(11, 9))

    # Use tab20 color map for nice colored regions
    contourf = ax.contourf(xx, yy, Z, levels=np.unique(Z).size, cmap="tab20", alpha=0.9)

    # Plot training samples
    for i in range(len(X)):
        x_pt, y_pt = X[i]
        cls = np.argmax(y[i])

        ax.scatter(
            x_pt, y_pt,
            c="white", s=280, edgecolors="black",
            linewidth=2, marker="o", zorder=10
        )
        ax.text(
            x_pt, y_pt - 0.12,
            f"C{cls}",
            ha="center", fontsize=9, fontweight="bold"
        )

    # Labels
    ax.set_title(
        "Multi-Class Decision Boundary (Your Neural Network)",
        fontsize=16, fontweight="bold"
    )
    ax.set_xlabel("x₁", fontsize=14)
    ax.set_ylabel("x₂", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Legend
    unique_classes = np.unique(Z)
    legend_elements = [Patch(label=f"Class {c}", facecolor="lightgray")]
    ax.legend(handles=legend_elements, fontsize=12, loc="upper right")

    plt.tight_layout()
    plt.show()

    print("=" * 70)
    print("Multi-class decision boundary visualization complete!")
    print("=" * 70)
