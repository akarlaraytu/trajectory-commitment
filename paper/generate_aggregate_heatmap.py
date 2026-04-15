#!/usr/bin/env python3
"""Generate a 2x2 panel of Cohen's d separation heatmaps from four prompt categories."""

import matplotlib.pyplot as plt
from pathlib import Path

# --- Paths ---
FIG_DIR = Path(__file__).resolve().parent.parent / "experiments" / "e07_trajectory_analysis" / "results" / "figures"
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Representative prompts per category
PANELS = [
    (9,  "(a) Factual"),
    (20, "(b) False Premise"),
    (42, "(c) Confabulation"),
    (57, "(d) Math"),
]

# --- Build figure ---
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

for ax, (idx, title) in zip(axes.flat, PANELS):
    img_path = FIG_DIR / f"03_separation_heatmap_{idx}.png"
    img = plt.imread(str(img_path))
    ax.imshow(img)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

fig.tight_layout(pad=1.5)

# --- Save ---
fig.savefig(OUT_DIR / "fig_heatmap_panel.pdf", bbox_inches="tight")
fig.savefig(OUT_DIR / "fig_heatmap_panel.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {OUT_DIR / 'fig_heatmap_panel.pdf'}")
print(f"Saved: {OUT_DIR / 'fig_heatmap_panel.png'}")
