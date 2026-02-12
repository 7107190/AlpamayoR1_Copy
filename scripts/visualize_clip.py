#!/usr/bin/env python3
"""
Visualize first clip with trajectory overlay
Creates a visualization showing:
1. Bird's eye view of trajectory
2. Velocity and acceleration profiles
3. Curvature over time
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("/scratch2/tropity24/nvidia_av_data")

# Load first clip
ego_files = sorted(list((DATA_DIR / "labels/egomotion").glob("*.parquet")))
first_ego = ego_files[0]
clip_id = first_ego.stem.replace(".egomotion", "")

print(f"Visualizing clip: {clip_id}")
print("=" * 80)

# Load ego motion data
df_ego = pd.read_parquet(first_ego)
print(f"Loaded {len(df_ego)} ego motion samples")

# Convert timestamp to seconds
df_ego['time_sec'] = (df_ego['timestamp'] - df_ego['timestamp'].min()) / 1e6

# Create figure with subplots
fig = plt.figure(figsize=(16, 10))

# 1. Bird's Eye View Trajectory
ax1 = plt.subplot(2, 3, 1)
scatter = ax1.scatter(df_ego['x'], df_ego['y'], c=df_ego['time_sec'],
                      cmap='viridis', s=10, alpha=0.7)
ax1.plot(df_ego['x'], df_ego['y'], 'b-', alpha=0.3, linewidth=0.5)
ax1.set_xlabel('X (meters)', fontsize=10)
ax1.set_ylabel('Y (meters)', fontsize=10)
ax1.set_title('Bird\'s Eye View Trajectory', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.axis('equal')
plt.colorbar(scatter, ax=ax1, label='Time (s)')

# 2. Velocity Profile
ax2 = plt.subplot(2, 3, 2)
v_magnitude = np.sqrt(df_ego['vx']**2 + df_ego['vy']**2 + df_ego['vz']**2)
ax2.plot(df_ego['time_sec'], v_magnitude, 'b-', linewidth=1.5)
ax2.set_xlabel('Time (seconds)', fontsize=10)
ax2.set_ylabel('Velocity (m/s)', fontsize=10)
ax2.set_title('Velocity Magnitude', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.fill_between(df_ego['time_sec'], 0, v_magnitude, alpha=0.2)

# 3. Acceleration Profile
ax3 = plt.subplot(2, 3, 3)
a_magnitude = np.sqrt(df_ego['ax']**2 + df_ego['ay']**2 + df_ego['az']**2)
ax3.plot(df_ego['time_sec'], a_magnitude, 'r-', linewidth=1.5, label='Total')
ax3.plot(df_ego['time_sec'], df_ego['ax'], 'g-', linewidth=1, alpha=0.7, label='ax')
ax3.plot(df_ego['time_sec'], df_ego['ay'], 'b-', linewidth=1, alpha=0.7, label='ay')
ax3.set_xlabel('Time (seconds)', fontsize=10)
ax3.set_ylabel('Acceleration (m/s²)', fontsize=10)
ax3.set_title('Acceleration Components', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=8)

# 4. Curvature
ax4 = plt.subplot(2, 3, 4)
ax4.plot(df_ego['time_sec'], df_ego['curvature'], 'purple', linewidth=1.5)
ax4.set_xlabel('Time (seconds)', fontsize=10)
ax4.set_ylabel('Curvature (1/m)', fontsize=10)
ax4.set_title('Path Curvature', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)

# 5. Velocity Components
ax5 = plt.subplot(2, 3, 5)
ax5.plot(df_ego['time_sec'], df_ego['vx'], 'r-', linewidth=1, label='vx (forward)')
ax5.plot(df_ego['time_sec'], df_ego['vy'], 'g-', linewidth=1, label='vy (lateral)')
ax5.plot(df_ego['time_sec'], df_ego['vz'], 'b-', linewidth=1, label='vz (vertical)')
ax5.set_xlabel('Time (seconds)', fontsize=10)
ax5.set_ylabel('Velocity (m/s)', fontsize=10)
ax5.set_title('Velocity Components (Ego Frame)', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.legend(fontsize=8)

# 6. Statistics Table
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

stats_text = f"""
Clip ID: {clip_id[:20]}...

Duration: {df_ego['time_sec'].max():.2f} seconds
Sample Rate: {len(df_ego) / df_ego['time_sec'].max():.1f} Hz
Total Samples: {len(df_ego)}

Velocity:
  Mean: {v_magnitude.mean():.2f} m/s
  Max: {v_magnitude.max():.2f} m/s
  Min: {v_magnitude.min():.2f} m/s

Acceleration:
  Mean: {a_magnitude.mean():.2f} m/s²
  Max: {a_magnitude.max():.2f} m/s²
  Std: {a_magnitude.std():.2f} m/s²

Curvature:
  Mean: {df_ego['curvature'].mean():.4f} 1/m
  Max: {df_ego['curvature'].max():.4f} 1/m
  Min: {df_ego['curvature'].min():.4f} 1/m

Distance Traveled: {np.sqrt(np.diff(df_ego['x'])**2 + np.diff(df_ego['y'])**2).sum():.1f} m
"""

ax6.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()

# Save figure
output_file = f"/scratch2/tropity24/nvidia_av_data/clip_{clip_id[:8]}_trajectory.png"
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved visualization to: {output_file}")

# Show plot
plt.show()

print("\n" + "=" * 80)
print("Visualization complete!")
print("=" * 80)
