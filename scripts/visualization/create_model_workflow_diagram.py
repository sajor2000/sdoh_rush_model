#!/usr/bin/env python3
"""
Create visual diagram showing proper model training workflow with calibration
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(5, 9.5, 'SDOH Model Training Workflow with Calibration', 
        fontsize=20, fontweight='bold', ha='center')
ax.text(5, 9.1, 'Following TRIPOD-AI Guidelines', 
        fontsize=14, ha='center', style='italic', color='gray')

# Colors
color_data = '#E8F4FD'
color_train = '#B8E0D2'
color_cal = '#FFE5B4'
color_test = '#FFB6C1'
color_model = '#D8BFD8'
color_final = '#98FB98'

# 1. Original Data
data_box = FancyBboxPatch((0.5, 7), 9, 1.2, 
                         boxstyle="round,pad=0.1",
                         facecolor=color_data, 
                         edgecolor='black', 
                         linewidth=2)
ax.add_patch(data_box)
ax.text(5, 7.6, 'Full Dataset (393,725 patients)', 
        fontsize=14, fontweight='bold', ha='center')
ax.text(5, 7.3, 'Prevalence: 6.6% with 2+ SDOH needs', 
        fontsize=11, ha='center')

# Arrow down
ax.arrow(5, 6.8, 0, -0.5, head_width=0.2, head_length=0.1, fc='black', ec='black')

# 2. Data Split
ax.text(5, 6.1, '60/20/20 Split (Random Seed: 2025)', 
        fontsize=12, ha='center', fontweight='bold')

# Three boxes for splits
# Training Set
train_box = FancyBboxPatch((0.5, 4.5), 2.8, 1.2,
                          boxstyle="round,pad=0.1",
                          facecolor=color_train,
                          edgecolor='darkgreen',
                          linewidth=2)
ax.add_patch(train_box)
ax.text(1.9, 5.4, 'Training Set', fontsize=12, fontweight='bold', ha='center')
ax.text(1.9, 5.1, '60% (236,235)', fontsize=10, ha='center')
ax.text(1.9, 4.8, 'For model fitting', fontsize=9, ha='center', style='italic')

# Calibration Set
cal_box = FancyBboxPatch((3.6, 4.5), 2.8, 1.2,
                        boxstyle="round,pad=0.1",
                        facecolor=color_cal,
                        edgecolor='darkorange',
                        linewidth=2)
ax.add_patch(cal_box)
ax.text(5, 5.4, 'Calibration Set', fontsize=12, fontweight='bold', ha='center')
ax.text(5, 5.1, '20% (78,745)', fontsize=10, ha='center')
ax.text(5, 4.8, 'For Platt scaling', fontsize=9, ha='center', style='italic')

# Test Set
test_box = FancyBboxPatch((6.7, 4.5), 2.8, 1.2,
                         boxstyle="round,pad=0.1",
                         facecolor=color_test,
                         edgecolor='darkred',
                         linewidth=2)
ax.add_patch(test_box)
ax.text(8.1, 5.4, 'Test Set', fontsize=12, fontweight='bold', ha='center')
ax.text(8.1, 5.1, '20% (78,745)', fontsize=10, ha='center')
ax.text(8.1, 4.8, 'Final evaluation', fontsize=9, ha='center', style='italic')

# Arrows from splits
ax.arrow(1.9, 4.4, 0, -0.5, head_width=0.15, head_length=0.1, fc='darkgreen', ec='darkgreen')
ax.arrow(5, 4.4, 0, -1.5, head_width=0.15, head_length=0.1, fc='darkorange', ec='darkorange')

# 3. Cross-validation box
cv_box = FancyBboxPatch((0.5, 2.5), 2.8, 1.3,
                       boxstyle="round,pad=0.1",
                       facecolor='#F0F0F0',
                       edgecolor='black',
                       linewidth=2,
                       linestyle='dashed')
ax.add_patch(cv_box)
ax.text(1.9, 3.5, '5-Fold Cross-Validation', fontsize=11, fontweight='bold', ha='center')
ax.text(1.9, 3.2, 'Optimize hyperparameters', fontsize=9, ha='center')
ax.text(1.9, 2.95, 'Score: 0.7×AUC + 0.3×(1-ECE)', fontsize=9, ha='center', style='italic')
ax.text(1.9, 2.7, 'Focus on calibration', fontsize=9, ha='center', color='red')

# Arrow to base model
ax.arrow(3.4, 3.1, 0.8, 0, head_width=0.15, head_length=0.1, fc='black', ec='black')

# 4. Base Model
base_model = FancyBboxPatch((4.5, 2.5), 2.5, 1.3,
                           boxstyle="round,pad=0.1",
                           facecolor=color_model,
                           edgecolor='purple',
                           linewidth=2)
ax.add_patch(base_model)
ax.text(5.75, 3.5, 'XGBoost Model', fontsize=12, fontweight='bold', ha='center')
ax.text(5.75, 3.2, 'Best parameters', fontsize=9, ha='center')
ax.text(5.75, 2.95, 'max_depth: 3-5', fontsize=9, ha='center')
ax.text(5.75, 2.7, 'Not calibrated yet', fontsize=9, ha='center', style='italic', color='red')

# Arrow down to calibration
ax.arrow(5.75, 2.4, 0, -0.5, head_width=0.15, head_length=0.1, fc='darkorange', ec='darkorange')

# 5. Calibration Process
cal_process = FancyBboxPatch((4, 0.8), 3.5, 1.2,
                            boxstyle="round,pad=0.1",
                            facecolor='#FFFACD',
                            edgecolor='darkorange',
                            linewidth=2)
ax.add_patch(cal_process)
ax.text(5.75, 1.6, 'Platt Scaling Calibration', fontsize=12, fontweight='bold', ha='center')
ax.text(5.75, 1.3, 'Fit sigmoid on calibration set', fontsize=9, ha='center')
ax.text(5.75, 1.0, 'Prevents overfitting', fontsize=9, ha='center', style='italic')

# Arrow to final model
ax.arrow(7.6, 1.4, 0.5, 0, head_width=0.15, head_length=0.1, fc='black', ec='black')

# 6. Final Model
final_model = FancyBboxPatch((8.2, 0.8), 1.6, 1.2,
                            boxstyle="round,pad=0.1",
                            facecolor=color_final,
                            edgecolor='green',
                            linewidth=3)
ax.add_patch(final_model)
ax.text(9, 1.6, 'Calibrated', fontsize=11, fontweight='bold', ha='center')
ax.text(9, 1.35, 'Model', fontsize=11, fontweight='bold', ha='center')
ax.text(9, 1.0, 'ECE < 0.05', fontsize=9, ha='center', color='green', fontweight='bold')

# Test evaluation arrow
ax.annotate('', xy=(8.1, 2.5), xytext=(8.1, 4.4),
            arrowprops=dict(arrowstyle='->', lw=2, color='darkred'))
ax.text(8.3, 3.5, 'Evaluate', fontsize=10, rotation=-90, va='center', color='darkred')

# Add key metrics box
metrics_box = FancyBboxPatch((0.2, 0.2), 3.5, 0.4,
                            boxstyle="round,pad=0.05",
                            facecolor='#F5F5F5',
                            edgecolor='gray',
                            linewidth=1)
ax.add_patch(metrics_box)
ax.text(1.95, 0.4, 'Key Metrics: AUC: 0.76 | ECE: <0.05 | Brier: 0.06', 
        fontsize=10, ha='center', fontweight='bold')

# Add legend
ax.text(0.2, 6.5, 'Key Points:', fontsize=12, fontweight='bold')
ax.text(0.2, 6.2, '• Separate calibration set prevents overfitting', fontsize=10)
ax.text(0.2, 5.9, '• Cross-validation optimizes for calibration + discrimination', fontsize=10)
ax.text(0.2, 5.6, '• Test set provides unbiased final evaluation', fontsize=10)
ax.text(0.2, 5.3, '• Platt scaling ensures trustworthy probabilities', fontsize=10)

# Save the figure
plt.tight_layout()
plt.savefig('results/figures/model_training_workflow.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('model_training_workflow.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')

print("✅ Model training workflow diagram created!")
print("   Saved to: results/figures/model_training_workflow.png")

# Show the plot
plt.show()