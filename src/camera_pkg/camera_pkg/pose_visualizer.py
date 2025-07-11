import cv2
import numpy as np
from pupil_apriltags import Detector as apriltag

# ----------------------------
# Camera intrinsics (example values)
# Replace with your real camera calibration
fx = 600
fy = 600
cx = 320
cy = 240
camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0,  0,  1]])
camera_params = (fx, fy, cx, cy)
tag_size = 0.05  # in meters (adjust to your tag)

# ----------------------------
# Setup
detector = apriltag.Detector()
cap = cv2.VideoCapture(0)  # Change if not default webcam

def draw_axes(img, camera_matrix, rvec, tvec, length=0.03):
    axis = np.float32([[length, 0, 0],
                       [0, length, 0],
                       [0, 0, length]]).reshape(-1, 3)
    origin = np.float32([[0, 0, 0]]).reshape(-1, 3)

    imgpts, _ = cv2.projectPoints(np.vstack((origin, axis)), rvec, tvec, camera_matrix, distCoeffs=None)

    origin = tuple(imgpts[0].ravel().astype(int))
    x_axis = tuple(imgpts[1].ravel().astype(int))
    y_axis = tuple(imgpts[2].ravel().astype(int))
    z_axis = tuple(imgpts[3].ravel().astype(int))

    cv2.line(img, origin, x_axis, (0, 0, 255), 2)  # X - red
    cv2.line(img, origin, y_axis, (0, 255, 0), 2)  # Y - green
    cv2.line(img, origin, z_axis, (255, 0, 0), 2)  # Z - blue

# ----------------------------
# Main loop
img_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tags = detector.detect(gray)

    for tag in tags:
        corners = np.array(tag.corners, dtype=np.float32)
        center = tuple(np.mean(corners, axis=0).astype(int))
        cv2.polylines(frame, [np.int32(corners)], True, (0, 255, 255), 2)
        cv2.putText(frame, f"ID: {tag.tag_id}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Pose estimation
        obj_pts = np.array([
            [-tag_size/2, -tag_size/2, 0],
            [ tag_size/2, -tag_size/2, 0],
            [ tag_size/2,  tag_size/2, 0],
            [-tag_size/2,  tag_size/2, 0]
        ], dtype=np.float32)
        img_pts = corners.reshape(-1, 2)

        success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, camera_matrix, None)
        if success:
            draw_axes(frame, camera_matrix, rvec, tvec)

    cv2.imshow("AprilTag Pose Viewer", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        filename = f"capture_{img_id:03d}.png"
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")
        img_id += 1

cap.release()
cv2.destroyAllWindows()

