# Matplotlib Unsearch Style - light and dark mode
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

# Define the custom style dictionary for light mode
UNSEARCH_COLORS_LIGHT = {
    "keppel": "#3fa79bff",
    "hunyadi_yellow": "#edae49ff",
    "amaranth": "#d1495bff",
    "alice_blue": "#e8eef2ff",
    "onyx": "#37393aff",
    "pink": "#c09bd8",
}

# Define the custom style dictionary for dark mode
UNSEARCH_COLORS_DARK = {
    "keppel": "#3fa79bff",
    "hunyadi_yellow": "#edae49ff",
    "amaranth": "#d1495bff",
    "onyx": "#37393aff",
    "alice_blue": "#e8eef2ff",
    "pink": "#c09bd8",
    # "light_gray": "white",
}

# Define the custom style dictionary for light mode
UNSEARCH_MPL_STYLE_LIGHT = {
    "axes.facecolor": "white",
    "axes.edgecolor": UNSEARCH_COLORS_LIGHT["onyx"],
    "axes.labelcolor": UNSEARCH_COLORS_LIGHT["keppel"],
    "figure.facecolor": UNSEARCH_COLORS_LIGHT["alice_blue"],
    "figure.edgecolor": UNSEARCH_COLORS_LIGHT["alice_blue"],
    "savefig.facecolor": UNSEARCH_COLORS_LIGHT["alice_blue"],
    "savefig.edgecolor": UNSEARCH_COLORS_LIGHT["alice_blue"],
    "text.color": UNSEARCH_COLORS_LIGHT["onyx"],
    "xtick.color": UNSEARCH_COLORS_LIGHT["onyx"],
    "axes.labelcolor": UNSEARCH_COLORS_LIGHT["onyx"],
    "ytick.color": UNSEARCH_COLORS_LIGHT["onyx"],
    "grid.color": UNSEARCH_COLORS_LIGHT["hunyadi_yellow"],
    "grid.linestyle": "--",
    "patch.edgecolor": UNSEARCH_COLORS_LIGHT["amaranth"],
    "patch.facecolor": UNSEARCH_COLORS_LIGHT["keppel"],
    "axes.prop_cycle": cycler(
        color=[
            UNSEARCH_COLORS_LIGHT["keppel"],
            UNSEARCH_COLORS_LIGHT["amaranth"],
            UNSEARCH_COLORS_LIGHT["hunyadi_yellow"],
            # UNSEARCH_COLORS_LIGHT["alice_blue"],
            UNSEARCH_COLORS_LIGHT["onyx"],
            UNSEARCH_COLORS_LIGHT["pink"],
        ]
    ),
}

# Define the custom style dictionary for dark mode
UNSEARCH_MPL_STYLE_DARK = {
    "axes.facecolor": UNSEARCH_COLORS_DARK["onyx"],
    "axes.edgecolor": UNSEARCH_COLORS_DARK["alice_blue"],
    "axes.labelcolor": UNSEARCH_COLORS_DARK["alice_blue"],
    "figure.facecolor": UNSEARCH_COLORS_DARK["onyx"],
    "figure.edgecolor": UNSEARCH_COLORS_DARK["onyx"],
    "savefig.facecolor": UNSEARCH_COLORS_DARK["onyx"],
    "savefig.edgecolor": UNSEARCH_COLORS_DARK["onyx"],
    "text.color": UNSEARCH_COLORS_DARK["alice_blue"],
    "xtick.color": UNSEARCH_COLORS_DARK["alice_blue"],
    "axes.labelcolor": UNSEARCH_COLORS_DARK["alice_blue"],
    "ytick.color": UNSEARCH_COLORS_DARK["alice_blue"],
    "grid.color": UNSEARCH_COLORS_DARK["hunyadi_yellow"],
    "grid.linestyle": "--",
    "patch.edgecolor": UNSEARCH_COLORS_DARK["hunyadi_yellow"],
    "patch.facecolor": UNSEARCH_COLORS_DARK["amaranth"],
    "axes.prop_cycle": cycler(
        color=[
            UNSEARCH_COLORS_DARK["keppel"],
            UNSEARCH_COLORS_DARK["amaranth"],
            UNSEARCH_COLORS_DARK["hunyadi_yellow"],
            UNSEARCH_COLORS_DARK["alice_blue"],
            # UNSEARCH_COLORS_DARK["onyx"],
            UNSEARCH_COLORS_DARK["pink"],
        ]
    ),
}

# Colormaps for light mode
UNSEARCH_CMAPS_LIGHT = {
    "unsearch_divergent": LinearSegmentedColormap.from_list(
        "unsearch_divergent",
        [
            (0, UNSEARCH_COLORS_LIGHT["amaranth"]),
            (0.5, UNSEARCH_COLORS_LIGHT["alice_blue"]),
            (1, UNSEARCH_COLORS_LIGHT["keppel"]),
        ],
    ),
    "unsearch_sequential": LinearSegmentedColormap.from_list(
        "unsearch_sequential", [(0, "white"), (1, UNSEARCH_COLORS_LIGHT["keppel"])]
    ),
}

# Colormaps for dark mode
UNSEARCH_CMAPS_DARK = {
    "unsearch_divergent": LinearSegmentedColormap.from_list(
        "unsearch_divergent",
        [
            (0, UNSEARCH_COLORS_DARK["amaranth"]),
            (0.5, UNSEARCH_COLORS_DARK["onyx"]),
            (1, UNSEARCH_COLORS_DARK["keppel"]),
        ],
    ),
    "unsearch_sequential": LinearSegmentedColormap.from_list(
        "unsearch_sequential",
        [
            (0, UNSEARCH_COLORS_DARK["amaranth"]),
            (1, UNSEARCH_COLORS_DARK["alice_blue"]),
        ],
    ),
    "unsearch_sequential_r": LinearSegmentedColormap.from_list(
        "unsearch_sequential_r",
        [
            (0, UNSEARCH_COLORS_DARK["alice_blue"]),
            (1, UNSEARCH_COLORS_DARK["amaranth"]),
        ],
    ),
}


def set_style(dark_mode=False):
    """Set the custom style."""
    if dark_mode:
        plt.style.use(UNSEARCH_MPL_STYLE_DARK)
        for name, cmap in UNSEARCH_CMAPS_DARK.items():
            try:
                cm.register_cmap(name=name, cmap=cmap)
            except ValueError:
                # cmap already registered
                pass
    else:
        plt.style.use(UNSEARCH_MPL_STYLE_LIGHT)
        for name, cmap in UNSEARCH_CMAPS_LIGHT.items():
            try:
                cm.register_cmap(name=name, cmap=cmap)
            except ValueError:
                # cmap already registered
                pass
