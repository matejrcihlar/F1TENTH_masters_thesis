import numpy as np
import matplotlib.pyplot as plt

# === User settings ===
N = 700  # Number of points to remove from start
M = 400   # Number of points to remove from end
CSV_FILE = '/home/mrc/plots/poses.csv'
OUTPUT_PNG = '/home/mrc/plots/pose_plots.svg'
# === Load CSV ===
data = np.loadtxt(CSV_FILE, delimiter=',', skiprows=1)
time = data[:, 0]
x1 = data[:, 1]
y1 = data[:, 2]
x2 = data[:, 3]
y2 = data[:, 4]

# === Trim data ===
def trim(arr):
    if N + M >= len(arr):
        raise ValueError("N + M must be smaller than the number of data points.")
    return arr[N:len(arr)-M]

time = trim(time)
time = [i-time[0] for i in time]
x1 = trim(x1)
y1 = trim(y1)
x2 = trim(x2)
y2 = trim(y2)

# === Plotting ===
fig, axs = plt.subplots(3, 1, figsize=(6, 9))

# 1. XY Path
axs[0].plot(x1, y1, label='Agent')
axs[0].plot(x2, y2, label='Opponent')
k1 = 80  # arrow spacing
k2 = 290
x1_triangle = x1[650:1500]
y1_triangle = y1[650:1500]
dx = np.diff(x1_triangle)
dy = np.diff(y1_triangle)
lengths = np.sqrt(dx**2 + dy**2)
nonzero = lengths > 1e-6
dx[nonzero] /= lengths[nonzero]
dy[nonzero] /= lengths[nonzero]
indices = np.arange(0, len(dx), k1)
axs[0].quiver(
    x1_triangle[indices], y1_triangle[indices],
    dx[indices], dy[indices],
    angles='xy', scale_units='xy', scale=5,
    width=0.005, headwidth=6, headlength=5,
    color='blue', alpha=1
)
x2_triangle = x2[100:1200]
y2_triangle = y2[100:1200]
dx = np.diff(x2_triangle)
dy = np.diff(y2_triangle)
lengths = np.sqrt(dx**2 + dy**2)
nonzero = lengths > 1e-6
dx[nonzero] /= lengths[nonzero]
dy[nonzero] /= lengths[nonzero]
indices = np.arange(0, len(dx), k2)
axs[0].quiver(
    x2_triangle[indices], y2_triangle[indices],
    dx[indices], dy[indices],
    angles='xy', scale_units='xy', scale=5,
    width=0.005, headwidth=6, headlength=5,
    color='orange', alpha=1
)
axs[0].set_title('XY Path')
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')
axs[0].legend()
axs[0].axis('equal')

# 2. X over Time
axs[1].plot(time, x1, label='Agent')
axs[1].plot(time, x2, label='Opponent')
axs[1].set_title('X over Time')
axs[1].set_xlabel('Time [s]')
axs[1].set_ylabel('X')
axs[1].legend()

# 3. Y over Time
axs[2].plot(time, y1, label='Agent')
axs[2].plot(time, y2, label='Opponent')
axs[2].set_title('Y over Time')
axs[2].set_xlabel('Time [s]')
axs[2].set_ylabel('Y')
axs[2].legend()

plt.tight_layout()
plt.savefig(OUTPUT_PNG, format='svg')