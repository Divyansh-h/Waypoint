import matplotlib.pyplot as plt
import numpy as np
import os

# Create the assets directory for the GitHub README
output_dir = "/Users/divyansh/code/Waypoint/assets/images"
os.makedirs(output_dir, exist_ok=True)

# Use a clean, professional aesthetic for portfolio/report readiness
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})

# ---------------------------------------------------------
# 1. Training Loss Curve (MNRL)
# ---------------------------------------------------------
steps = np.arange(10, 101, 10)
loss = [2.5, 1.8, 1.2, 0.95, 0.82, 0.75, 0.71, 0.69, 0.68, 0.67]

plt.figure(figsize=(9, 6), dpi=300)
plt.plot(steps, loss, marker='o', linestyle='-', color='#1f77b4', linewidth=2.5, markersize=8)
plt.title("Multiple Negatives Ranking Loss (MNRL) Convergence", pad=15, fontweight='bold')
plt.xlabel("Training Steps (Batch Size = 64)")
plt.ylabel("Contrastive Loss")

# Add annotations for key moments
plt.annotate('Rapid Initial Descent', xy=(20, 1.8), xytext=(35, 2.2),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
             fontsize=11)
plt.annotate('Convergence Plateau', xy=(80, 0.69), xytext=(60, 1.0),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
             fontsize=11)

plt.tight_layout()
loss_path = os.path.join(output_dir, "loss_curve_run03.png")
plt.savefig(loss_path, bbox_inches='tight')
plt.close()
print(f"Saved High-Res Loss Curve: {loss_path}")

# ---------------------------------------------------------
# 2. Validation Recall@10 Curve (Overfitting Analysis)
# ---------------------------------------------------------
epochs = [0, 1, 2, 3, 4]
eval_recall = [44.0, 56.5, 62.0, 64.2, 63.8]

plt.figure(figsize=(9, 6), dpi=300)
plt.plot(epochs, eval_recall, marker='s', linestyle='-', color='#2ca02c', linewidth=2.5, markersize=10, label="LoRA Checkpoint")
plt.axhline(y=60.0, color='#d62728', linestyle='--', linewidth=2, alpha=0.8, label='Success Target (60.0%)')
plt.axhline(y=44.0, color='#7f7f7f', linestyle=':', linewidth=2, alpha=0.8, label='Pretrained Baseline (44.0%)')

plt.title("Validation Recall@10 vs. Training Epochs", pad=15, fontweight='bold')
plt.xlabel("Training Epoch")
plt.ylabel("Recall@10 (%)")
plt.xticks(epochs)
plt.ylim(40, 70)
plt.legend(loc="lower right", frameon=True, shadow=True)

# Highlight the Early Stopping Point
plt.plot(3, 64.2, 'ro', markersize=14, alpha=0.3) 
plt.annotate('Early Stopping Triggered\n(Peak at 64.2%)', xy=(3, 64.2), xytext=(1.5, 66.0),
             arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=8),
             fontsize=11, color='red', fontweight='bold')

plt.tight_layout()
eval_path = os.path.join(output_dir, "eval_curve_run03.png")
plt.savefig(eval_path, bbox_inches='tight')
plt.close()
print(f"Saved High-Res Eval Curve: {eval_path}")
