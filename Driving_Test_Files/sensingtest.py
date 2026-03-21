import depthai as dai
import numpy as np
import cv2
import threading
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
BLOB_PATH   = Path(r"C:\Users\teamten\py_39\detect\runs\oakd\exp16\weights/best_openvino_2022.1_6shave.blob") #Or the path to your blob
CLASSES     = ["Raisin Box"] #Make sure this matches your model, check data.yaml for this info
NUM_CLASSES = len(CLASSES)
IMG_SIZE    = 416
CONF_THRES  = 0.5
IOU_THRES   = 0.4

COLORS = [
    (0,   200, 255),
    (0,   255, 100),
    (255, 100,   0),
    (180,   0, 255),
    (0,   100, 255),
]

# ── YOLOv8 OUTPUT PARSER ──────────────────────────────────────────────────────
def parse_yolov8(raw, conf_thres, iou_thres, img_size, nc):
    data = np.array(raw, dtype=np.float32).flatten()
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
            cx, cy, w, h = cx * img_size, cy * img_size, w * img_size, h * img_size
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
    return [(boxes[i], scores[i], class_ids[i]) for i in idxs.flatten()]


# ── SPATIAL LOOKUP ────────────────────────────────────────────────────────────
def get_spatial_coords(depth_frame, box, img_size):
    dh, dw = depth_frame.shape[:2]
    x, y, w, h = box
    dx1 = int(x * dw / img_size)
    dy1 = int(y * dh / img_size)
    dx2 = int((x + w) * dw / img_size)
    dy2 = int((y + h) * dh / img_size)
    cx  = (dx1 + dx2) // 2
    cy  = (dy1 + dy2) // 2
    rw  = max(1, (dx2 - dx1) // 5)
    rh  = max(1, (dy2 - dy1) // 5)
    roi = depth_frame[max(0, cy-rh):min(dh, cy+rh), max(0, cx-rw):min(dw, cx+rw)]
    valid = roi[roi > 0]
    if valid.size == 0:
        return None
    Z  = float(np.median(valid))

    # Calibrated intrinsics from device (416x416 resolution)
    FX = 317.3552
    FY = 317.3552
    CX = 220.6296
    CY = 199.9118

    X  = (cx - CX) * Z / FX
    Y  = (cy - CY) * Z / FY
    dist = float(np.sqrt(X**2 + Y**2 + Z**2))  # Euclidean distance in mm
    return X, Y, Z, dist


# ── DRAW DETECTIONS ───────────────────────────────────────────────────────────
def draw_detections(frame, detections, depth_frame=None):
    for (box, conf, cls_id) in detections:
        x, y, w, h = box
        color = COLORS[cls_id % len(COLORS)]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.drawMarker(frame, (x + w // 2, y + h // 2), color,
                       cv2.MARKER_CROSS, 12, 1, cv2.LINE_AA)
        label = f"{CLASSES[cls_id]} {conf:.2f}"
        label_y = max(y - 5, 36)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x, label_y - th - 6), (x + tw + 6, label_y), color, -1)
        cv2.putText(frame, label, (x + 3, label_y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        if depth_frame is not None:
            coords = get_spatial_coords(depth_frame, box, IMG_SIZE)
            if coords:
                X, Y, Z, dist = coords
                spatial = f"X:{X/1000:+.2f}m Y:{Y/1000:+.2f}m Z:{Z/1000:.2f}m"
                dist_label = f"Dist: {dist/1000:.2f}m"
                cv2.putText(frame, spatial, (x, y + h + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
                cv2.putText(frame, dist_label, (x, y + h + 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, dist_label, (x, y + h + 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    return frame


# ── BUILD PIPELINE ────────────────────────────────────────────────────────────
pipeline = dai.Pipeline()

# RGB camera (CAM_A default)
cam = pipeline.create(dai.node.Camera)
cam.build()
cam_out = cam.requestOutput((IMG_SIZE, IMG_SIZE), dai.ImgFrame.Type.BGR888p)

# Left mono (CAM_B)
left = pipeline.create(dai.node.Camera)
left.build(dai.CameraBoardSocket.CAM_B)
left_out = left.requestOutput((640, 400), dai.ImgFrame.Type.GRAY8)

# Right mono (CAM_C)
right = pipeline.create(dai.node.Camera)
right.build(dai.CameraBoardSocket.CAM_C)
right_out = right.requestOutput((640, 400), dai.ImgFrame.Type.GRAY8)

# Stereo depth — MEDIAN_OFF avoids SIPP memory crash entirely
stereo = pipeline.create(dai.node.StereoDepth)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_DENSITY)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
stereo.setOutputSize(IMG_SIZE, IMG_SIZE)
stereo.initialConfig.setMedianFilter(dai.MedianFilter.MEDIAN_OFF)

left_out.link(stereo.left)
right_out.link(stereo.right)

# Neural network
nn = pipeline.create(dai.node.NeuralNetwork)
nn.setBlobPath(BLOB_PATH)
nn.setNumInferenceThreads(2)
nn.setNumNCEPerInferenceThread(1)
nn.input.setBlocking(False)
cam_out.link(nn.input)

# Output queues
q_rgb   = cam_out.createOutputQueue(maxSize=4, blocking=False)
q_nn    = nn.out.createOutputQueue(maxSize=4, blocking=False)
q_depth = stereo.depth.createOutputQueue(maxSize=4, blocking=False)

# ── START PIPELINE ────────────────────────────────────────────────────────────
pipeline_thread = threading.Thread(target=pipeline.run, daemon=True)
pipeline_thread.start()

print("OAK-D Lite — Spatial Trash Detector (DepthAI 3.x)")
print(f"  Blob   : {BLOB_PATH}")
print(f"  Classes: {', '.join(CLASSES)}")
print("  Depth  : enabled — X/Y/Z in metres")
print("Press  Q  to quit.\n")

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
fps_counter = 0
fps_display = 0.0
fps_timer   = cv2.getTickCount()

try:
    while True:
        frame_msg = q_rgb.get()
        nn_msg    = q_nn.get()
        depth_msg = q_depth.tryGet()

        frame = frame_msg.getCvFrame()
        if frame.shape[0] != IMG_SIZE or frame.shape[1] != IMG_SIZE:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

        depth_frame = depth_msg.getFrame() if depth_msg is not None else None

        raw   = nn_msg.getFirstTensor().flatten()
        dets  = parse_yolov8(raw, CONF_THRES, IOU_THRES, IMG_SIZE, NUM_CLASSES)
        frame = draw_detections(frame, dets, depth_frame)

        if depth_frame is not None:
            depth_vis = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            for (box, _, _) in dets:
                bx, by, bw, bh = box
                cv2.rectangle(depth_vis, (bx, by), (bx + bw, by + bh), (255, 255, 255), 1)
            cv2.imshow("Depth View", depth_vis)

        if dets and depth_frame is not None:
            for (box, conf, cls_id) in dets:
                coords = get_spatial_coords(depth_frame, box, IMG_SIZE)
                if coords:
                    X, Y, Z, dist = coords
                    print(f"  {CLASSES[cls_id]:10s} conf={conf:.2f}  "
                          f"X={X/1000:+.3f}m  Y={Y/1000:+.3f}m  Z={Z/1000:.3f}m  "
                          f"dist={dist/1000:.3f}m")

        fps_counter += 1
        elapsed = (cv2.getTickCount() - fps_timer) / cv2.getTickFrequency()
        if elapsed >= 1.0:
            fps_display = fps_counter / elapsed
            fps_counter = 0
            fps_timer   = cv2.getTickCount()

        cv2.putText(frame, f"FPS: {fps_display:.1f}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Detections: {len(dets)}", (8, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("OAK-D Lite — Spatial Trash Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Done.")
