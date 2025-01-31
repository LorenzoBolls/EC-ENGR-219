import itertools
import numpy as np
import matplotlib.pyplot as plt

def plot_mat(mat, xticklabels=None, yticklabels=None, pic_fname=None, 
             size=(-1,-1), if_show_values=True, colorbar=True, grid='k', 
             xlabel=None, ylabel=None, title=None, vmin=None, vmax=None):
    if size == (-1, -1):
        size = (mat.shape[1] / 3, mat.shape[0] / 3)

    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(1,1,1)

    # Create the pcolor plot
    im = ax.pcolor(mat, cmap=plt.cm.Blues, linestyle='-', 
                   linewidth=0.5, edgecolor=grid, vmin=vmin, vmax=vmax)

    if colorbar:
        plt.colorbar(im, fraction=0.046, pad=0.25)

    # Set ticks
    lda_num_topics = mat.shape[0]
    nmf_num_topics = mat.shape[1]
    yticks = np.arange(lda_num_topics)
    xticks = np.arange(nmf_num_topics)
    ax.set_xticks(xticks + 0.5)
    ax.set_yticks(yticks + 0.5)
    if xticklabels is None:
        xticklabels = [str(i) for i in xticks]
    if yticklabels is None:
        yticklabels = [str(i) for i in yticks]
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)

    # Tick labels on both left and right
    ax.tick_params(labelright=True, labeltop=False)

    # Axis labels
    if ylabel:
        plt.ylabel(ylabel, fontsize=15)
    if xlabel:
        plt.xlabel(xlabel, fontsize=15)
    if title:
        plt.title(title, fontsize=15)

    # Invert y-axis for visual top-down alignment
    ax.invert_yaxis()

    # Function to show values inside the cells
    def show_values(pc, fmt="%.2f", **kw):
        pc.update_scalarmappable()
        ax = pc.axes
        for p, color, value in itertools.zip_longest(pc.get_paths(), pc.get_facecolors(), pc.get_array().flatten()):
            # Skip masked array values
            if np.ma.is_masked(value):
                continue

            x, y = p.vertices[:-2, :].mean(0)
            # Determine appropriate text color (black/white) based on background
            if np.all(color[:3] > 0.5):
                text_color = (0.0, 0.0, 0.0)
            else:
                text_color = (1.0, 1.0, 1.0)
            
            ax.text(
                x, y, fmt % value,
                ha="center", va="center",
                color=text_color, **kw, fontsize=10
            )

    if if_show_values:
        show_values(im, fmt="%.0f")  # for integer-like labeling, use "%.0f" or "%d" with int-casting

    plt.tight_layout()
    if pic_fname:
        plt.savefig(pic_fname, dpi=300, transparent=True)
    plt.show()
    plt.close()
