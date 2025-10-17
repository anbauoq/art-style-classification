import math
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_class_distribution(class_counts, title):
    """
    Generates and displays a horizontal bar chart for class distribution.

    Args:
        class_counts (pd.Series or pd.DataFrame): 
            A pandas Series with class names as the index and counts as values,
            or a DataFrame with 'Class Name' and 'Count' columns.
        title (str): 
            The title for the plot.
    """
    # Ensure the input is a DataFrame in the correct format
    if isinstance(class_counts, pd.Series):
        dist_df = class_counts.reset_index()
        dist_df.columns = ['Class Name', 'Count']
    elif isinstance(class_counts, pd.DataFrame) and all(col in class_counts.columns for col in ['Class Name', 'Count']):
        dist_df = class_counts.copy()
    else:
        raise ValueError("Input must be a pandas Series or a DataFrame with 'Class Name' and 'Count' columns.")
        
    dist_df = dist_df.sort_values(by='Count', ascending=False)

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    # Dynamically adjust height based on the number of classes
    num_classes = len(dist_df)
    fig_height = max(5, num_classes * 0.5) 
    plt.figure(figsize=(10, fig_height))

    ax = sns.barplot(
        x='Count',
        y='Class Name',
        data=dist_df,
        palette='viridis',
        orient='h',
        hue='Class Name',
        legend=False
    )

    # Add the exact count labels to the end of each bar
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', padding=5, fontsize=10)

    # --- Set titles and labels ---
    plt.title(title, fontsize=18, pad=20)
    plt.xlabel('Image Count', fontsize=12)
    plt.ylabel('Art Style Class', fontsize=12)
    plt.xlim(0, dist_df['Count'].max() * 1.15)  # Set x-limit for label space
    
    plt.tight_layout()
    plt.show()

def plot_rgb_kde_by_class(
    df,
    class_col: str = "class",
    channels: tuple = ("R", "G", "B"),
    cols: int = 4,
    xlim: tuple = (0, 255),
    figsize_per_row: float = 3.0,
    suptitle: str = "RGB Color Intensity Distributions per Class",
    palette: dict | None = None,
    alpha: float = 0.3,
    fill: bool = True,
    bw_adjust: float | None = None,
    return_objects: bool = False,
):
    """
    Plot per-class KDEs for channels in a grid with normalized y-limits.

    If return_objects is True, returns (fig, axes); otherwise returns None.
    """

    try:
        import seaborn as sns
        use_sns = True
    except ImportError:
        use_sns = False

    classes = sorted(df[class_col].dropna().unique().tolist())
    if not classes:
        raise ValueError("No classes found to plot.")

    n = len(classes)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * figsize_per_row))
    if hasattr(axes, "flatten"):
        axes = axes.flatten()
    else:
        axes = [axes]

    # Default palette
    if palette is None:
        if set(channels) == {"R", "G", "B"}:
            palette = {"R": "red", "G": "green", "B": "blue"}
        else:
            palette = {}

    max_density = 0.0

    for i, cls in enumerate(classes):
        ax = axes[i]
        df_cls = df[df[class_col] == cls]
        for ch in channels:
            series = df_cls[ch].dropna()
            if series.empty:
                continue
            color = palette.get(ch, None)

            if use_sns:
                kwargs = dict(common_norm=False)
                if bw_adjust is not None:
                    kwargs["bw_adjust"] = bw_adjust
                sns.kdeplot(
                    series, ax=ax, label=str(ch),
                    fill=fill, alpha=alpha, color=color, **kwargs
                )
            else:
                ax.hist(
                    series, bins=50, density=True,
                    alpha=alpha, label=str(ch), color=color,
                    edgecolor="black", linewidth=0.3
                )

        ax.set_title(str(cls), fontsize=12)
        ax.set_xlim(*xlim)
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Density")
        ax.legend(loc="upper right", fontsize=8)

        max_density = max(max_density, ax.get_ylim()[1])

    # Normalize y-limits across populated axes
    for ax in axes[:n]:
        ax.set_ylim(0, max_density if max_density > 0 else ax.get_ylim()[1])

    # Remove empty subplots
    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(suptitle, fontsize=18, y=1.02)
    fig.tight_layout()

    # Show and suppress repr unless requested
    plt.show()
    if return_objects:
        return fig, axes

def plot_curves(history):
    """Plots the training and validation loss and macro-F1 curves."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curve
    plt.figure()
    plt.plot(epochs, history['train_loss'], label="Train Loss")
    plt.plot(epochs, history['val_loss'], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Over Epochs")
    plt.legend()
    plt.show()
    
    # F-1 curve
    plt.figure()
    plt.plot(epochs, history['train_f1'], label="Train F1")
    plt.plot(epochs, history['val_f1'], label="Val F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("Macro-F1 Over Epochs")
    plt.legend()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plots a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title("Confusion Matrix (Counts)")
    plt.tight_layout()
    plt.show()


def plot_metric_learning_curves(history):
    """
    Plots training loss and validation performance for metric learning.
    Creates two subplots:
    1. Training Loss (e.g., Triplet Loss)
    2. Validation Metrics (e.g., k-NN F1-score and Accuracy)
    """
    if 'train_loss' not in history or not history['train_loss']:
        print("Training loss not found in history. Cannot plot.")
        return

    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- Plot 1: Training Loss ---
    ax1.plot(epochs, history['train_loss'], 'o-', label='Train Loss')
    ax1.set_title('Training Loss per Epoch', fontsize=16)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_xticks(epochs)
    ax1.legend()

    # --- Plot 2: F1 Score Performance ---
    if 'train_f1' in history:
        ax2.plot(epochs, history['train_f1'], 'o-', label='Train Macro-F1')
        
    ax2.plot(epochs, history['val_f1'], 'o-', label='Validation Macro-F1')
    
    ax2.set_title('Train vs. Validation F1 Score', fontsize=16)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Macro-F1 Score', fontsize=12)
    ax2.set_xticks(epochs)
    ax2.legend()

    plt.tight_layout()
    plt.show()


def plot_embeddings_2d(df, x_col, y_col, hue_col, title, s=50, alpha=0.8):
    """
    Generates a scatter plot for 2D embeddings from a DataFrame.
    """
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=hue_col,
        palette='tab20',
        s=s,
        alpha=alpha
    )
    plt.title(title, fontsize=16)
    plt.xlabel('UMAP Component 1', fontsize=12)
    plt.ylabel('UMAP Component 2', fontsize=12)
    plt.legend(title=hue_col.replace('_', ' ').title(), bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_image_dimensions_distribution(
    df,
    width_col: str = "width",
    height_col: str = "height",
    bins: int = 30,
    figsize: tuple = (8, 4),
    suptitle: str = "Image Dimensions Distribution",
    kde: bool = True,
    width_color: str = "C0",
    height_color: str = "C1",
    alpha: float = 0.8,
):
    """
    Plot side-by-side histograms for image width and height with distinct colors.

    Returns:
    (fig, axes): matplotlib Figure and Axes array
    """
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns
        use_sns = True
    except ImportError:
        use_sns = False

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(suptitle, fontsize=16)

    w = df[width_col].dropna()
    h = df[height_col].dropna()

    if use_sns:
        sns.histplot(w, kde=kde, bins=bins, ax=axes[0], color=width_color, alpha=alpha)
        axes[0].set_title("Image Widths")

        sns.histplot(h, kde=kde, bins=bins, ax=axes[1], color=height_color, alpha=alpha)
        axes[1].set_title("Image Heights")
    else:
        axes[0].hist(w, bins=bins, color=width_color, alpha=alpha, edgecolor="black", linewidth=0.5)
        axes[0].set_title("Image Widths")

        axes[1].hist(h, bins=bins, color=height_color, alpha=alpha, edgecolor="black", linewidth=0.5)
        axes[1].set_title("Image Heights")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig, axes