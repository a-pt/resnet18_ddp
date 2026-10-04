import os
import matplotlib.pyplot as plt
import numpy as np

# Set dark/sleek style for TensorBoard publication quality look
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
fig_size = (8, 5)

out_dir = r'd:\Projects\resnet18_ddp\docs\images'
os.makedirs(out_dir, exist_ok=True)

epochs = np.array([1, 2, 3, 4, 5])

# Event data extracted from TensorBoard event logs
gpu1_train_loss = [1.4209, 0.9349, 0.7061, 0.5484, 0.4370]
gpu1_val_loss   = [1.4516, 0.9045, 0.7253, 0.6327, 0.4645]
gpu1_accuracy   = [52.33, 68.77, 75.04, 79.18, 84.15]
gpu1_time       = [7.79, 7.49, 7.73, 7.51, 7.51]

gpu2_train_loss = [1.5024, 1.0109, 0.7878, 0.6300, 0.5093]
gpu2_val_loss   = [1.2705, 0.9770, 0.7757, 0.7416, 0.5314]
gpu2_accuracy   = [54.66, 65.79, 72.45, 74.75, 81.69]
gpu2_time       = [4.65, 4.50, 4.37, 4.35, 4.35]

lr_schedule     = [0.0010, 0.0009045, 0.0006545, 0.0003455, 0.0000955]

# Colors
c_1gpu = '#1f77b4' # Deep Blue
c_2gpu = '#ff7f0e' # Energetic Orange

# 1. Training & Validation Loss Comparison
fig, ax = plt.subplots(figsize=fig_size, dpi=300)
ax.plot(epochs, gpu1_train_loss, 'o-', color=c_1gpu, label='1x RTX 4090 - Train Loss', linewidth=2.5, markersize=6)
ax.plot(epochs, gpu1_val_loss, 'o--', color=c_1gpu, label='1x RTX 4090 - Val Loss', linewidth=2, markersize=6, alpha=0.7)
ax.plot(epochs, gpu2_train_loss, 's-', color=c_2gpu, label='2x RTX 4090 (DDP) - Train Loss', linewidth=2.5, markersize=6)
ax.plot(epochs, gpu2_val_loss, 's--', color=c_2gpu, label='2x RTX 4090 (DDP) - Val Loss', linewidth=2, markersize=6, alpha=0.7)

ax.set_title('ResNet-18 Loss Curves (1x GPU vs 2x GPU DDP)', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Cross-Entropy Loss', fontsize=12)
ax.set_xticks(epochs)
ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'tensorboard_train_loss.png'))
plt.close(fig)

# 2. Validation Accuracy Comparison
fig, ax = plt.subplots(figsize=fig_size, dpi=300)
ax.plot(epochs, gpu1_accuracy, 'o-', color=c_1gpu, label='1x RTX 4090 (Batch 128)', linewidth=2.5, markersize=7)
ax.plot(epochs, gpu2_accuracy, 's-', color=c_2gpu, label='2x RTX 4090 DDP (Batch 256 Global)', linewidth=2.5, markersize=7)

ax.set_title('Validation Accuracy Progression (%)', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
ax.set_xticks(epochs)
ax.set_ylim(45, 90)
for i, (a1, a2) in enumerate(zip(gpu1_accuracy, gpu2_accuracy)):
    ax.annotate(f'{a1:.1f}%', (epochs[i], a1), textcoords="offset points", xytext=(0,8), ha='center', fontsize=9, color=c_1gpu, fontweight='bold')
    ax.annotate(f'{a2:.1f}%', (epochs[i], a2), textcoords="offset points", xytext=(0,-14), ha='center', fontsize=9, color=c_2gpu, fontweight='bold')

ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'tensorboard_val_accuracy.png'))
plt.close(fig)

# 3. Learning Rate & Epoch Execution Time
fig, ax1 = plt.subplots(figsize=fig_size, dpi=300)

ax1.plot(epochs, gpu1_time, 'o-', color=c_1gpu, label='1x RTX 4090 Epoch Time (s)', linewidth=2.5)
ax1.plot(epochs, gpu2_time, 's-', color=c_2gpu, label='2x RTX 4090 Epoch Time (s)', linewidth=2.5)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Epoch Training Time (seconds)', fontsize=12)
ax1.set_xticks(epochs)
ax1.set_ylim(0, 10)

ax2 = ax1.twinx()
ax2.plot(epochs, lr_schedule, 'k:', label='Cosine LR Schedule', linewidth=2, alpha=0.6)
ax2.set_ylabel('Learning Rate', fontsize=12, color='gray')
ax2.tick_params(axis='y', labelcolor='gray')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)

plt.title('Epoch Training Time & Learning Rate Schedule', fontsize=14, fontweight='bold', pad=12)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'tensorboard_learning_rate.png'))
plt.close(fig)

# 4. Speedup, Efficiency & Throughput Benchmarks
fig, (ax_tp, ax_sp) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

# Throughput
bars_tp = ax_tp.bar(['1x RTX 4090', '2x RTX 4090 (DDP)'], [6614, 11390], color=[c_1gpu, c_2gpu], width=0.5)
ax_tp.set_ylabel('Throughput (images/sec)', fontsize=11, fontweight='bold')
ax_tp.set_title('Training Throughput', fontsize=12, fontweight='bold')
ax_tp.set_ylim(0, 14000)
for bar in bars_tp:
    yval = bar.get_height()
    ax_tp.text(bar.get_x() + bar.get_width()/2.0, yval + 300, f'{yval:,} img/s', ha='center', va='bottom', fontweight='bold', fontsize=10)

# Speedup & Scaling Efficiency
bars_sp = ax_sp.bar(['Ideal 2x Scaling', 'Measured 2x DDP'], [2.0, 1.72], color=['#2ca02c', c_2gpu], width=0.5)
ax_sp.set_ylabel('Speedup Factor', fontsize=11, fontweight='bold')
ax_sp.set_title('Speedup & Scaling Efficiency (86.0%)', fontsize=12, fontweight='bold')
ax_sp.set_ylim(0, 2.4)
ax_sp.axhline(2.0, color='gray', linestyle='--', alpha=0.7)
ax_sp.text(0, 2.05, '100% Efficiency', ha='center', va='bottom', fontsize=9, color='gray')
ax_sp.text(1, 1.77, '1.72x Speedup (86.0%)', ha='center', va='bottom', fontweight='bold', fontsize=10, color=c_2gpu)

fig.suptitle('PyTorch DDP Performance Benchmarks on Dual RTX 4090', fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'tensorboard_scaling_benchmark.png'))
plt.close(fig)

print(f"Charts successfully exported to {out_dir}")
