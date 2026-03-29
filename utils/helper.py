import os
import matplotlib.pyplot as plt

def ensure_dirs(dirs):
    """Ensure that all directories in the list exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def plot_image(image, label):
    """Plot a single image and its corresponding label."""
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f"Label: {label}")
    plt.axis("off")
    plt.show()
