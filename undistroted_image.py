import cv2
import numpy as np
import math

data = np.load("camera_calibration.npz")
camera_matrix = data["camera_matrix"]
dist_coeffs = data["dist_coeffs"]

cap = cv2.VideoCapture(1)

points = []   # store clicked points
distance_px = None

# Mouse callback function
def mouse_click(event, x, y, flags, param):
    global points, distance_px

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

        # If 2 points clicked, calculate distance
        if len(points) == 2:
            x1, y1 = points[0]
            x2, y2 = points[1]

            distance_px = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            print(f"✅ Distance: {distance_px:.2f} pixels")

        # If more than 2 points, reset and start again
        if len(points) > 2:
            points = [(x, y)]
            distance_px = None


cv2.namedWindow("Undistorted")
cv2.setMouseCallback("Undistorted", mouse_click)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # Best new camera matrix
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )

    undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_mtx)

    # Optional crop
    x, y, w_roi, h_roi = roi
    undistorted = undistorted[y:y+h_roi, x:x+w_roi]

    display = undistorted.copy()

    # Draw points and line
    if len(points) >= 1:
        cv2.circle(display, points[0], 6, (0, 255, 0), -1)

    if len(points) >= 2:
        cv2.circle(display, points[1], 6, (0, 255, 0), -1)
        cv2.line(display, points[0], points[1], (255, 0, 0), 2)

        if distance_px is not None:
            mid_x = (points[0][0] + points[1][0]) // 2
            mid_y = (points[0][1] + points[1][1]) // 2
            cv2.putText(display, f"{distance_px:.2f}px", (mid_x, mid_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show
    cv2.imshow("Original", frame)
    cv2.imshow("Undistorted", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        points = []
        distance_px = None
        print("🔁 Reset points")

cap.release()
cv2.destroyAllWindows()
