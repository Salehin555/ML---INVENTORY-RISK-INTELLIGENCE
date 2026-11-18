"""Shared figure style: one colour-blind-safe palette, sans-serif type sized to the
caption (Times New Roman), recessive axes, panel letters, 600-dpi PDF output with embedded fonts."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# professional palette: sky blue, maroon, navy, gold, rose, purple (no green, no grey fills)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2e86c1", "#8e3b46", "#1b3a6b", "#c9a227", "#d98880", "#1b3a6b", "#6c3483", "#b03a2e")
INK, INK2, MUTED, GRID = "#0b0b0b", "#1b2a41", "#7f8fa6", "#e3eaf3"
NEUTRAL = "#85c1e9"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600, "figure.dpi": 150,
    "lines.linewidth": 1.4, "axes.titleweight": "bold", "axes.titlelocation": "left"})

FIGDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def panel(ax, letter, x=-0.02, y=1.04):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=8.5, fontweight="bold", color=INK, ha="right", va="bottom")


def save(fig, name):
    path = os.path.join(FIGDIR, name + ".pdf")
    fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return path
