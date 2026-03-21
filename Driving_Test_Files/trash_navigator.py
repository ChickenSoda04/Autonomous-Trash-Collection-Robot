#!/usr/bin/env python3
"""
trash_navigator.py
ROS2 node — OAK-D Myriad X inference + LD19 LiDAR stop.

Bandwidth optimisations:
  - Inference runs on Myriad X — only NN results cross USB, not full frames
  - No onnxruntime, no Pi CPU inference load
  - ImageManip converts video to BGR888p on-device before NN
  - maxSize=1 on all queues
  - USB2 mode — stable connection
  - Timer at 10Hz matches camera FPS
  - LiDAR reads 1 packet per loop
  - Single-threaded ROS executor
  - Preallocated numpy arrays
"""

import math
import time
import depthai
import numpy as np
import cv2
import serial
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
BLOB_PATH        = "/home/projects/best_simplified_openvino_2021.4_6shave.blob"  # Or the path to your blob
CLASSES          = ["Raisin Box"] #Make sure all these params match your model, check your data.yaml for this info
NUM_CLASSES      = len(CLASSES)
IMG_SIZE         = 416
CONF_THRES       = 0.5
IOU_THRES        = 0.4
NORM_FACTOR      = IMG_SIZE / 2.0

STOP_DISTANCE_M  = 0.16
LIDAR_PORT       = "/dev/ttyUSB0"
LIDAR_BAUD       = 230400
FORWARD_CONE     = 20.0
MIN_RANGE        = 0.15
MAX_RANGE        = 1.50
CLUSTER_DIST     = 0.08
MIN_CLUSTER_SIZE = 2
PACKETS_PER_LOOP = 1             # One LiDAR packet per loop — minimal USB traffic

KP            = 1.0
MAX_THROTTLE  = 0.2

HEADER      = 0x54
VER_LEN     = 0x2C
PACKET_SIZE = 47


# ── YOLOV8 PARSER ─────────────────────────────────────────────────────────────
def parse_yolov8(raw, conf_thres, iou_thres, img_size, nc):
    data        = np.asarray(raw, dtype=np.float32)
    total       = 4 + nc
    num_anchors = data.size // total
    data        = data.reshape(total, num_anchors).T
    normalised  = data[:, :4].max() <= 1.0

    boxes, scores, class_ids = [], [], []
    for row in data:
        cls_scores = row[4:]
        cls_id     = int(cls_scores.argmax())
        conf       = float(cls_scores[cls_id])
        if conf < conf_thres:
            continue
        cx, cy, w, h = row[:4]
        if normalised:
            cx, cy, w, h = (cx * img_size, cy * img_size,
                            w  * img_size, h  * img_size)
        x1 = max(0, int(cx - w / 2))
        y1 = max(0, int(cy - h / 2))
        x2 = min(img_size - 1, int(cx + w / 2))
        y2 = min(img_size - 1, int(cy + h / 2))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(conf)
        class_ids.append(cls_id)

    if not boxes:
        return []
    idxs = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)
    if len(idxs) == 0:
        return []
    idxs = idxs.flatten() if hasattr(idxs, 'flatten') else idxs
    return [(boxes[i], scores[i], class_ids[i]) for i in idxs]


# ── LIDAR HELPERS ─────────────────────────────────────────────────────────────
def lidar_read_packet(ser):
    try:
        b1 = ser.read(1)
        if not b1 or b1[0] != HEADER:
            return None
        b2 = ser.read(1)
        if not b2 or b2[0] != VER_LEN:
            return None
        rest = ser.read(PACKET_SIZE - 2)
        if len(rest) != PACKET_SIZE - 2:
            return None
        return b1 + b2 + rest
    except Exception:
        return None


def lidar_parse_packet(packet):
    start_angle = ((packet[5] << 8) | packet[4]) / 100.0
    end_angle   = ((packet[43] << 8) | packet[42]) / 100.0
    if end_angle < start_angle:
        end_angle += 360.0
    pts = []
    for i in range(12):
        off     = 6 + 3 * i
        dist_mm = (packet[off + 1] << 8) | packet[off]
        if dist_mm == 0:
            continue
        dist_m = dist_mm / 1000.0
        if dist_m < MIN_RANGE or dist_m > MAX_RANGE:
            continue
        angle_deg = start_angle + (end_angle - start_angle) * i / 11.0
        angle_deg = angle_deg % 360.0
        if not (angle_deg <= FORWARD_CONE or angle_deg >= (360.0 - FORWARD_CONE)):
            continue
        angle_rad = math.radians(angle_deg)
        x = dist_m * math.cos(angle_rad)
        y = dist_m * math.sin(angle_rad)
        pts.append((x, y, dist_m))
    return pts


def lidar_cluster(points):
    if not points:
        return []
    clusters = []
    current  = [points[0]]
    for i in range(1, len(points)):
        if math.hypot(points[i][0] - points[i-1][0],
                      points[i][1] - points[i-1][1]) <= CLUSTER_DIST:
            current.append(points[i])
        else:
            clusters.append(current)
            current = [points[i]]
    clusters.append(current)
    return [c for c in clusters if len(c) >= MIN_CLUSTER_SIZE]


def lidar_centroid_distance(clusters):
    best = MAX_RANGE
    for c in clusters:
        x = sum(p[0] for p in c) / len(c)
        y = sum(p[1] for p in c) / len(c)
        best = min(best, math.hypot(x, y))
    return best


# ── ROS2 NODE ─────────────────────────────────────────────────────────────────
class TrashNavigator(Node):

    def __init__(self):
        super().__init__('trash_navigator_node')
        self.publisher    = self.create_publisher(Twist, '/cmd_vel', 10)
        self.stopped      = False
        self.current_dist = MAX_RANGE

        self._init_oakd()
        self._init_lidar()

        # 10Hz matches camera FPS — no wasted loop iterations
        self.create_timer(0.1, self.navigation_loop)
        self.get_logger().info('TrashNavigator ready.')

    # ── OAK-D + Myriad X inference ────────────────────────────────────────────
    def _init_oakd(self):
        self._device = None
        self._q_nn   = None

        devices = depthai.Device.getAllAvailableDevices()
        if not devices:
            self.get_logger().error('No OAK-D device found.')
            return

        device = devices[0]
        if str(device.state) != 'XLinkDeviceState.X_LINK_UNBOOTED':
            self.get_logger().warn(
                'OAK-D not UNBOOTED: {}'.format(device.state))
            return

        pipeline = depthai.Pipeline()

        cam_rgb = pipeline.create(depthai.node.ColorCamera)
        cam_rgb.setResolution(
            depthai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setVideoSize(IMG_SIZE, IMG_SIZE)
        cam_rgb.setColorOrder(depthai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setInterleaved(False)
        cam_rgb.setFps(10)

        # Convert to BGR888p on-device before NN — no frame data crosses USB
        manip = pipeline.create(depthai.node.ImageManip)
        manip.initialConfig.setFrameType(depthai.RawImgFrame.Type.BGR888p)
        manip.setMaxOutputFrameSize(IMG_SIZE * IMG_SIZE * 3)
        cam_rgb.video.link(manip.inputImage)

        # Inference runs entirely on Myriad X
        nn = pipeline.create(depthai.node.NeuralNetwork)
        nn.setBlobPath(BLOB_PATH)
        nn.setNumInferenceThreads(2)
        nn.setNumNCEPerInferenceThread(1)
        nn.input.setBlocking(False)
        manip.out.link(nn.input)

        # Only NN results cross USB — tiny compared to full frames
        xout_nn = pipeline.create(depthai.node.XLinkOut)
        xout_nn.setStreamName("nn")
        nn.out.link(xout_nn.input)

        self._device = depthai.Device(pipeline, device, True)
        self._q_nn   = self._device.getOutputQueue(
            "nn", maxSize=1, blocking=False)
        self.get_logger().info('OAK-D Myriad X inference started.')

    # ── LiDAR init ────────────────────────────────────────────────────────────
    def _init_lidar(self):
        try:
            self._ser = serial.Serial(LIDAR_PORT, LIDAR_BAUD, timeout=0.005)
            self.get_logger().info('LiDAR opened on {}'.format(LIDAR_PORT))
        except serial.SerialException as e:
            self.get_logger().error('LiDAR error: {}'.format(e))
            self._ser = None

    # ── LiDAR distance ────────────────────────────────────────────────────────
    def _get_lidar_distance(self):
        if self._ser is None:
            return self.current_dist
        pts = []
        for _ in range(PACKETS_PER_LOOP):
            packet = lidar_read_packet(self._ser)
            if packet:
                pts.extend(lidar_parse_packet(packet))
        if not pts:
            return self.current_dist
        pts_sorted = sorted(pts, key=lambda p: math.atan2(p[1], p[0]))
        clusters   = lidar_cluster(pts_sorted)
        if not clusters:
            return self.current_dist
        return lidar_centroid_distance(clusters)

    # ── Main navigation loop ──────────────────────────────────────────────────
    def navigation_loop(self):
        twist = Twist()

        if self.stopped:
            self.publisher.publish(twist)
            return

        # LiDAR stop check
        self.current_dist = self._get_lidar_distance()
        if self.current_dist <= STOP_DISTANCE_M:
            self.stopped = True
            self.publisher.publish(twist)
            self.get_logger().warn(
                'Target reached at {:.3f}m — stopping.'.format(
                    self.current_dist))
            return

        if self._q_nn is None:
            return

        # Read NN result — only detection data, no frame pixels cross USB
        try:
            nn_msg = self._q_nn.tryGet()
        except RuntimeError as e:
            self.get_logger().warn(
                'OAK-D error — attempting reopen: {}'.format(str(e)[:60]))
            try:
                self._device.close()
            except Exception:
                pass
            self._device = None
            self._q_nn   = None
            time.sleep(5.0)
            self._init_oakd()
            return

        if nn_msg is None:
            return

        raw  = nn_msg.getFirstLayerFp16()
        dets = parse_yolov8(raw, CONF_THRES, IOU_THRES, IMG_SIZE, NUM_CLASSES)

        if not dets:
            self.get_logger().info('No detection — holding.')
            return

        best_box, best_conf, best_cls = max(dets, key=lambda d: d[1])
        bx, by, bw, bh = best_box

        obj_cx   = bx + bw / 2.0
        error_x  = (obj_cx - NORM_FACTOR) / NORM_FACTOR

        twist.linear.x  = float(MAX_THROTTLE)
        twist.angular.z = float(KP * -error_x)
        self.publisher.publish(twist)

        self.get_logger().info(
            '{} conf={:.2f}  err={:+.3f}  dist={:.3f}m'.format(
                CLASSES[best_cls], best_conf, error_x, self.current_dist))

    def destroy_node(self):
        try:
            self._device.close()
        except Exception:
            pass
        try:
            if self._ser and self._ser.is_open:
                self._ser.close()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = TrashNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
