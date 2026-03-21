import matplotlib
matplotlib.use('Qt5Agg')  # Switch to 'TkAgg' or 'QtAgg' if this fails

import math
import numpy as np
import serial
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PORT = "COM6" # Or your COM port
BAUD = 230400
HEADER = 0x54
VER_LEN = 0x2C
PACKET_SIZE = 47
MIN_RANGE = 0.15
MAX_RANGE = 1.50
CLUSTER_DISTANCE = 0.08
MIN_CLUSTER_SIZE = 2          # Lowered from 3 — fewer points needed to form a cluster

FORWARD_CONE_DEG = 20.0

# Fewer retained points = old data drops off much faster
MAX_POINTS_ON_SCREEN = 100

# More packets drained per frame = faster visual update
PACKETS_PER_FRAME = 10

ser = serial.Serial(PORT, BAUD, timeout=0.005)  # Very short timeout
recent_points = []

def read_packet():
    while True:
        b1 = ser.read(1)
        if not b1 or b1[0] != HEADER:
            continue
        b2 = ser.read(1)
        if not b2 or b2[0] != VER_LEN:
            continue
        rest = ser.read(PACKET_SIZE - 2)
        if len(rest) != PACKET_SIZE - 2:
            continue
        return b1 + b2 + rest

def parse_packet(packet):
    start_angle = ((packet[5] << 8) | packet[4]) / 100.0
    end_angle   = ((packet[43] << 8) | packet[42]) / 100.0
    if end_angle < start_angle:
        end_angle += 360.0
    pts = []
    for i in range(12):
        off = 6 + 3 * i
        dist_mm = (packet[off + 1] << 8) | packet[off]
        if dist_mm == 0:
            continue
        dist_m = dist_mm / 1000.0
        if dist_m < MIN_RANGE or dist_m > MAX_RANGE:
            continue
        angle_deg = start_angle + (end_angle - start_angle) * i / 11.0
        angle_deg = angle_deg % 360.0
        a = angle_deg
        if not (a <= FORWARD_CONE_DEG or a >= (360.0 - FORWARD_CONE_DEG)):
            continue
        angle_rad = math.radians(angle_deg)
        x = dist_m * math.cos(angle_rad)
        y = dist_m * math.sin(angle_rad)
        pts.append((x, y, dist_m, angle_deg))
    return pts

def euclidean_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def cluster_points(points, threshold=CLUSTER_DISTANCE):
    if not points:
        return []
    clusters = []
    current_cluster = [points[0]]
    for i in range(1, len(points)):
        if euclidean_distance(points[i - 1], points[i]) <= threshold:
            current_cluster.append(points[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [points[i]]
    clusters.append(current_cluster)
    return clusters

def compute_centroid(cluster):
    x_avg = sum(p[0] for p in cluster) / len(cluster)
    y_avg = sum(p[1] for p in cluster) / len(cluster)
    d_avg = math.hypot(x_avg, y_avg)
    return (x_avg, y_avg, d_avg)

EMPTY = np.empty((0, 2))

# ── Radar figure ──────────────────────────────────────────────────────────────
plt.ion()
fig = plt.figure(figsize=(8, 8), facecolor='#0a0a0a')
ax  = fig.add_subplot(111, facecolor='#0a0a0a')
ax.set_aspect('equal')
ax.set_xlim(-MAX_RANGE - 0.1, MAX_RANGE + 0.1)
ax.set_ylim(-MAX_RANGE - 0.1, MAX_RANGE + 0.1)
ax.axis('off')

for r in [0.25, 0.50, 0.75, 1.00, 1.25, 1.50]:
    ax.add_patch(plt.Circle((0, 0), r, color='#1a3a1a', fill=False, linewidth=0.8))
    ax.text(0, r + 0.02, f'{r:.2f}m', color='#2a6a2a', fontsize=7, ha='center', va='bottom')

for deg in range(0, 360, 30):
    rad = math.radians(deg)
    ax.plot([0, MAX_RANGE * math.cos(rad)], [0, MAX_RANGE * math.sin(rad)],
            color='#1a3a1a', linewidth=0.6)
    lx = (MAX_RANGE + 0.08) * math.cos(rad)
    ly = (MAX_RANGE + 0.08) * math.sin(rad)
    ax.text(lx, ly, f'{deg}°', color='#2a6a2a', fontsize=7, ha='center', va='center')

ax.add_patch(mpatches.Wedge((0, 0), MAX_RANGE + 0.1,
                             -FORWARD_CONE_DEG, FORWARD_CONE_DEG,
                             color='#002800', alpha=0.6))

for sign in [-1, 1]:
    rad = math.radians(sign * FORWARD_CONE_DEG)
    ax.plot([0, (MAX_RANGE + 0.1) * math.cos(rad)],
            [0, (MAX_RANGE + 0.1) * math.sin(rad)],
            color='#00aa00', linewidth=1.0, linestyle='--')

ax.plot(0, 0, 'o', color='#00ff41', markersize=5)
ax.annotate('', xy=(0.18, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='#00ff41', lw=1.5))

scatter_fwd  = ax.scatter([], [], s=8,   color='#00ff41', alpha=0.85, zorder=3)
scatter_cent = ax.scatter([], [], s=120, color='#ff4444', marker='+',
                          linewidths=2.5, zorder=5)

ax.text(-MAX_RANGE - 0.08, MAX_RANGE + 0.05,
        'LD19 Live Scan', color='#00ff41', fontsize=11, fontweight='bold', va='top')
dist_text  = ax.text(-MAX_RANGE - 0.08, MAX_RANGE - 0.05,
                     'Forward: --', color='#ff4444',
                     fontsize=14, fontweight='bold', va='top')
stats_text = ax.text(MAX_RANGE - 0.05, MAX_RANGE + 0.05,
                     '', color='#00ff41', fontsize=8, va='top', ha='right')
ax.text(0, -(MAX_RANGE + 0.08), f'Cone: ±{FORWARD_CONE_DEG}°',
        color='#2a6a2a', fontsize=8, ha='center', va='top')

plt.tight_layout(pad=0.3)
plt.pause(0.1)

print(f"Window open — cone: ±{FORWARD_CONE_DEG}°\n")

frame_count = 0

try:
    while True:
        pts_this_frame = []
        for _ in range(PACKETS_PER_FRAME):
            try:
                packet = read_packet()
                pts_this_frame.extend(parse_packet(packet))
            except Exception:
                break

        if not pts_this_frame:
            plt.pause(0.001)
            continue

        frame_count += 1

        # Replace buffer entirely each frame rather than appending and trimming —
        # this means only the freshest points are ever shown
        recent_points.clear()
        recent_points.extend(pts_this_frame[-MAX_POINTS_ON_SCREEN:])

        forward_centroid = None

        if recent_points:
            fwd_xy   = [(p[0], p[1]) for p in recent_points]
            asorted  = sorted(recent_points, key=lambda p: math.atan2(p[1], p[0]))
            axy      = [(p[0], p[1]) for p in asorted]
            clusters = [c for c in cluster_points(axy) if len(c) >= MIN_CLUSTER_SIZE]

            if clusters:
                centroids        = [compute_centroid(c) for c in clusters]
                forward_centroid = min(centroids, key=lambda c: c[2])
                dist_m           = forward_centroid[2]
                print(f"Object centroid: {dist_m:.3f} m  "
                      f"(x={forward_centroid[0]:.3f}, y={forward_centroid[1]:.3f})")
                dist_text.set_text(f'Forward: {dist_m:.3f} m')
            else:
                dist_text.set_text('Forward: no object')

            scatter_fwd.set_offsets(np.array(fwd_xy))
        else:
            scatter_fwd.set_offsets(EMPTY)
            dist_text.set_text('Forward: no object')

        scatter_cent.set_offsets(
            np.array([[forward_centroid[0], forward_centroid[1]]])
            if forward_centroid is not None else EMPTY
        )

        stats_text.set_text(f'Points: {len(recent_points)}\nFrames: {frame_count}')

        plt.pause(0.001)

except KeyboardInterrupt:
    print("\nStopped.")
finally:
    ser.close()
