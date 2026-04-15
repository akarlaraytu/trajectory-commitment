"""
Generate paper figures:
1. Conceptual bifurcation diagram (Fig 1)
2. KL vs step curve (aggregate across bifurcating prompts)
3. Bifurcation rates bar chart (cleaned up)
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

fig_dir = Path(__file__).parent / "figures"
fig_dir.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# FIGURE 1: Conceptual Bifurcation Diagram
# ═══════════════════════════════════════════════════════════════

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.set_xlim(-3, 3)
ax.set_ylim(-0.5, 6.5)
ax.set_aspect('equal')
ax.axis('off')

# Shared trunk (steps -1 to 0)
trunk_x = [0, 0]
trunk_y = [6, 4]
ax.plot(trunk_x, trunk_y, '-', color='#333333', linewidth=3, solid_capstyle='round')

# Bifurcation point
ax.plot(0, 4, 'o', color='#333333', markersize=12, zorder=5)
ax.annotate('$\\mathbf{h}_0$\n(shared state)', xy=(0, 4), xytext=(1.6, 4.5),
            fontsize=10, ha='left', va='center',
            arrowprops=dict(arrowstyle='->', color='#666666', lw=1.2),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0', edgecolor='#999999'))

# Correct trajectory
corr_x = [0, -0.3, -0.8, -1.2, -1.5, -1.7]
corr_y = [4, 3.2, 2.4, 1.6, 0.8, 0.0]
ax.plot(corr_x, corr_y, '-', color='#2ca02c', linewidth=2.5, solid_capstyle='round')
for i, (cx, cy) in enumerate(zip(corr_x[1:], corr_y[1:])):
    ax.plot(cx, cy, 'o', color='#2ca02c', markersize=6, zorder=4)
ax.annotate('Correct\ntrajectory', xy=(-1.7, 0.0), xytext=(-2.5, 0.4),
            fontsize=10, ha='center', color='#2ca02c', fontweight='bold')

# Hallucinated trajectory
hall_x = [0, 0.3, 0.8, 1.2, 1.5, 1.7]
hall_y = [4, 3.2, 2.4, 1.6, 0.8, 0.0]
ax.plot(hall_x, hall_y, '-', color='#d62728', linewidth=2.5, solid_capstyle='round')
for i, (hx, hy) in enumerate(zip(hall_x[1:], hall_y[1:])):
    ax.plot(hx, hy, 'o', color='#d62728', markersize=6, zorder=4)
ax.annotate('Hallucinated\ntrajectory', xy=(1.7, 0.0), xytext=(2.5, 0.4),
            fontsize=10, ha='center', color='#d62728', fontweight='bold')

# Step labels on right side
step_labels = ['step 0\n(prompt)', 'step 1', 'step 2', 'step 3', 'step 4', 'step 5']
for i, (y, label) in enumerate(zip([4, 3.2, 2.4, 1.6, 0.8, 0.0], step_labels)):
    ax.text(-2.9, y, label, fontsize=8, color='#888888', ha='left', va='center',
            fontstyle='italic')

# Commitment window bracket
bracket_x = 2.7
ax.annotate('', xy=(bracket_x, 4.0), xytext=(bracket_x, 2.4),
            arrowprops=dict(arrowstyle='<->', color='#e67e22', lw=2))
ax.text(bracket_x + 0.15, 3.2, 'commitment\nwindow', fontsize=9,
        color='#e67e22', fontweight='bold', ha='left', va='center')

# Prompt input arrow
ax.annotate('', xy=(0, 6), xytext=(0, 6.4),
            arrowprops=dict(arrowstyle='->', color='#333333', lw=2))
ax.text(0, 6.5, 'prompt $x$', fontsize=11, ha='center', va='bottom',
        fontweight='bold')

# KL annotation
ax.annotate('$D_{\\mathrm{KL}} = 0$', xy=(0, 4), xytext=(-1.8, 3.6),
            fontsize=9, color='#555555',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#aaaaaa'))
ax.annotate('$D_{\\mathrm{KL}} \\gg 0$', xy=(0, 2.8), xytext=(-1.8, 2.0),
            fontsize=9, color='#555555',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#aaaaaa'))

# Token sampling annotation
ax.annotate('token\nsampling', xy=(0, 3.6), xytext=(0.8, 3.0),
            fontsize=8, color='#8e44ad', fontstyle='italic',
            arrowprops=dict(arrowstyle='->', color='#8e44ad', lw=1, connectionstyle='arc3,rad=0.2'))

plt.tight_layout()
plt.savefig(fig_dir / "fig1_conceptual_bifurcation.png", dpi=300, bbox_inches='tight')
plt.savefig(fig_dir / "fig1_conceptual_bifurcation.pdf", bbox_inches='tight')
plt.close()
print("Fig 1: conceptual bifurcation diagram saved")


# ═══════════════════════════════════════════════════════════════
# FIGURE: KL vs Step Curve (aggregate)
# ═══════════════════════════════════════════════════════════════

# Load bifurcation data
bif_path = Path(__file__).parent.parent / "experiments" / "e07_trajectory_analysis" / "results" / "bifurcation_Qwen_Qwen2.5-1.5B.json"
with open(bif_path) as f:
    bif_data = json.load(f)

# Extract KL per step from trajectory summary
traj = bif_data.get("trajectory_summary", {})
if traj:
    all_kl_curves = []
    for idx, td in traj.items():
        kl = td["kl_per_step"]
        all_kl_curves.append(kl)

    # Pad to same length
    max_len = max(len(k) for k in all_kl_curves)
    padded = []
    for k in all_kl_curves:
        padded.append(k + [np.nan] * (max_len - len(k)))
    kl_matrix = np.array(padded)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    # Individual curves (thin, transparent)
    for i in range(len(kl_matrix)):
        valid = ~np.isnan(kl_matrix[i])
        steps = np.arange(max_len)[valid]
        vals = kl_matrix[i][valid]
        ax.plot(steps, vals, '-', color='#cccccc', linewidth=0.8, alpha=0.6)

    # Median and IQR
    median_kl = np.nanmedian(kl_matrix, axis=0)
    q25 = np.nanpercentile(kl_matrix, 25, axis=0)
    q75 = np.nanpercentile(kl_matrix, 75, axis=0)
    steps = np.arange(max_len)

    ax.fill_between(steps, q25, q75, alpha=0.25, color='#2c3e50', label='IQR')
    ax.plot(steps, median_kl, '-o', color='#2c3e50', linewidth=2.5, markersize=6,
            label='Median', zorder=5)

    # Annotations
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.4, linewidth=1)
    ax.text(max_len - 1, 0.7, 'onset threshold (0.5)', fontsize=8,
            color='red', alpha=0.6, ha='right')

    # Step 0 annotation
    ax.annotate('$D_{\\mathrm{KL}}^{(0)} = 0$\n(identical logits)',
                xy=(0, 0), xytext=(1.5, median_kl[2] * 0.3),
                fontsize=8.5, ha='center',
                arrowprops=dict(arrowstyle='->', color='#666666', lw=1),
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#ffffcc', edgecolor='#cccc00'))

    ax.set_xlabel('Generation Step', fontsize=11)
    ax.set_ylabel('$D_{\\mathrm{KL}}(P_{\\mathrm{hall}} \\| P_{\\mathrm{corr}})$', fontsize=11)
    ax.set_title('Step-wise KL Divergence Across Bifurcating Prompts', fontsize=11)
    ax.legend(fontsize=9, loc='upper left')
    ax.set_xlim(-0.3, max_len - 0.7)
    ax.grid(True, alpha=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(fig_dir / "fig_kl_vs_step.png", dpi=300, bbox_inches='tight')
    plt.savefig(fig_dir / "fig_kl_vs_step.pdf", bbox_inches='tight')
    plt.close()
    print("KL vs step curve saved")

# ═══════════════════════════════════════════════════════════════
# FIGURE: Bifurcation Rates (cleaned up for paper)
# ═══════════════════════════════════════════════════════════════

prompt_results = bif_data["prompt_results"]

fig, ax = plt.subplots(1, 1, figsize=(10, 3.5))

categories = []
for p in prompt_results:
    categories.append(p["category"])

cat_order = ["factual", "false_premise", "confabulation", "leading", "multi_hop", "math"]
cat_colors = {
    "factual": "#3498db",
    "false_premise": "#e74c3c",
    "confabulation": "#9b59b6",
    "leading": "#f39c12",
    "multi_hop": "#2ecc71",
    "math": "#1abc9c"
}

for i, p in enumerate(prompt_results):
    c_rate = p["correct_rate"]
    h_rate = p["hallucination_rate"]
    color = cat_colors.get(p["category"], "gray")

    ax.bar(i, c_rate, color=color, alpha=0.7, width=0.8)
    ax.bar(i, -h_rate, color=color, alpha=0.4, width=0.8)

    if p["is_bifurcating"]:
        ax.plot(i, max(c_rate, 0.02) + 0.05, '*', color='#f39c12',
                markersize=8, zorder=5)

ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_xlabel('Prompt Index', fontsize=10)
ax.set_ylabel('Rate', fontsize=10)
ax.set_ylim(-1.1, 1.1)

# Legend
legend_elements = [
    mpatches.Patch(facecolor=cat_colors[c], alpha=0.7, label=c.replace('_', ' '))
    for c in cat_order
]
legend_elements.append(plt.Line2D([0], [0], marker='*', color='w',
                                   markerfacecolor='#f39c12', markersize=10,
                                   label='Bifurcating'))
ax.legend(handles=legend_elements, fontsize=7.5, ncol=4, loc='upper center',
          bbox_to_anchor=(0.5, 1.15))

# Category boundaries
boundaries = []
prev_cat = prompt_results[0]["category"]
for i, p in enumerate(prompt_results):
    if p["category"] != prev_cat:
        boundaries.append(i - 0.5)
        prev_cat = p["category"]
for b in boundaries:
    ax.axvline(x=b, color='#cccccc', linewidth=0.5, linestyle='-')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.1, axis='y')

plt.tight_layout()
plt.savefig(fig_dir / "fig_bifurcation_rates.png", dpi=300, bbox_inches='tight')
plt.savefig(fig_dir / "fig_bifurcation_rates.pdf", bbox_inches='tight')
plt.close()
print("Bifurcation rates saved")


# ═══════════════════════════════════════════════════════════════
# FIGURE: Recovery Cost (window size vs recovery rate)
# Clean version of window patching for paper
# ═══════════════════════════════════════════════════════════════

patch_path = Path(__file__).parent.parent / "experiments" / "e07_trajectory_analysis" / "results" / "patching_Qwen_Qwen2.5-1.5B.json"
with open(patch_path) as f:
    patch_data = json.load(f)

window = patch_data["window_results"]
controls = patch_data["controls"]
baseline = controls["baseline_nopatch"]["correct_rate"]
random_ctrl = controls["random_clean"]["flip_rate"]

fig, ax = plt.subplots(1, 1, figsize=(5.5, 4))

wins = sorted(window.keys(), key=int)
h2c = [window[w]["h2c_rate"] for w in wins]
c2h = [window[w]["c2h_rate"] for w in wins]
win_labels = [int(w) for w in wins]

ax.plot(win_labels, h2c, '-s', color='#2ca02c', linewidth=2.5, markersize=9,
        label='H$\\to$C (correction)', zorder=5)
ax.plot(win_labels, c2h, '-s', color='#d62728', linewidth=2.5, markersize=9,
        label='C$\\to$H (corruption)', zorder=5)

ax.axhline(y=baseline, color='#7f7f7f', linestyle=':', linewidth=1.5,
           label=f'Baseline ({baseline:.1%})', alpha=0.7)
ax.axhline(y=random_ctrl, color='#bcbd22', linestyle='--', linewidth=1.5,
           label=f'Random control ({random_ctrl:.1%})', alpha=0.7)

ax.fill_between(win_labels, baseline, h2c, alpha=0.1, color='#2ca02c')

ax.set_xlabel('Patch Window Size (steps)', fontsize=11)
ax.set_ylabel('Flip Rate', fontsize=11)
ax.set_xticks(win_labels)
ax.set_xticklabels([f'[1..{w}]' for w in win_labels])
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0, 0.85)

# Annotation for recovery cost
ax.annotate('Recovery requires\nsustained intervention',
            xy=(4, h2c[-1]), xytext=(2.5, 0.5),
            fontsize=9, ha='center', color='#2ca02c', fontstyle='italic',
            arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.2))

plt.tight_layout()
plt.savefig(fig_dir / "fig_recovery_cost.png", dpi=300, bbox_inches='tight')
plt.savefig(fig_dir / "fig_recovery_cost.pdf", bbox_inches='tight')
plt.close()
print("Recovery cost saved")


# ═══════════════════════════════════════════════════════════════
# FIGURE: Layer Sweep (cleaned up for paper)
# ═══════════════════════════════════════════════════════════════

layer_sweep = patch_data["layer_sweep"]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True,
                                gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.08})

layers = sorted(layer_sweep.keys(), key=int)
h2c_rates = [layer_sweep[l]["h2c_rate"] for l in layers]
c2h_rates = [layer_sweep[l]["c2h_rate"] for l in layers]
layer_ints = [int(l) for l in layers]

# H→C (correction)
bars1 = ax1.bar(layer_ints, h2c_rates, color='#2ca02c', alpha=0.75, width=0.8)
ax1.axhline(y=random_ctrl, color='#bcbd22', linestyle='--', linewidth=1.2, alpha=0.7,
            label=f'Random ctrl ({random_ctrl:.0%})')
ax1.axhline(y=baseline, color='#7f7f7f', linestyle=':', linewidth=1.2, alpha=0.7,
            label=f'Baseline ({baseline:.0%})')

# Highlight best
best_h2c_idx = np.argmax(h2c_rates)
bars1[best_h2c_idx].set_edgecolor('#1a7a1a')
bars1[best_h2c_idx].set_linewidth(2)
ax1.annotate(f'L{layer_ints[best_h2c_idx]}: {h2c_rates[best_h2c_idx]:.0%}',
             xy=(layer_ints[best_h2c_idx], h2c_rates[best_h2c_idx]),
             xytext=(layer_ints[best_h2c_idx] - 5, h2c_rates[best_h2c_idx] + 0.05),
             fontsize=8.5, fontweight='bold', color='#1a7a1a',
             arrowprops=dict(arrowstyle='->', color='#1a7a1a', lw=1))

ax1.set_ylabel('H$\\to$C Flip Rate', fontsize=10)
ax1.set_ylim(0, 0.45)
ax1.legend(fontsize=8, loc='upper left')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(True, alpha=0.1, axis='y')
ax1.set_title('Activation Patching Layer Sweep (step = 1, $n$ = 24 per layer)', fontsize=11)

# C→H (corruption)
bars2 = ax2.bar(layer_ints, c2h_rates, color='#d62728', alpha=0.75, width=0.8)
best_c2h_idx = np.argmax(c2h_rates)
bars2[best_c2h_idx].set_edgecolor('#8b0000')
bars2[best_c2h_idx].set_linewidth(2)
ax2.annotate(f'L{layer_ints[best_c2h_idx]}: {c2h_rates[best_c2h_idx]:.0%}',
             xy=(layer_ints[best_c2h_idx], c2h_rates[best_c2h_idx]),
             xytext=(layer_ints[best_c2h_idx] - 5, c2h_rates[best_c2h_idx] + 0.05),
             fontsize=8.5, fontweight='bold', color='#8b0000',
             arrowprops=dict(arrowstyle='->', color='#8b0000', lw=1))

ax2.set_xlabel('Layer', fontsize=10)
ax2.set_ylabel('C$\\to$H Flip Rate', fontsize=10)
ax2.set_ylim(0, 1.0)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(True, alpha=0.1, axis='y')

plt.tight_layout()
plt.savefig(fig_dir / "fig_layer_sweep.png", dpi=300, bbox_inches='tight')
plt.savefig(fig_dir / "fig_layer_sweep.pdf", bbox_inches='tight')
plt.close()
print("Layer sweep saved")

print("\nAll figures generated.")
