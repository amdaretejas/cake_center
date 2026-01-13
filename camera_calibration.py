import cv2
import numpy as np
import glob
import os

# ---------- SETTINGS ----------
CHESSBOARD_SIZE = (8, 6)   # inner corners (columns, rows)  <-- change this
SQUARE_SIZE = 35.0         # in mm (optional, but good), you can keep it 1.0 too
IMAGE_FOLDER = "calib_imgs"  # folder where chessboard images are stored
SAVE_FILE = "camera_calibration.npz"
# -----------------------------

# Prepare object points (0,0,0), (1,0,0), (2,0,0)... scaled by square size
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []  # 3D points in real world
imgpoints = []  # 2D points in image plane

images = glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg")) + glob.glob(os.path.join(IMAGE_FOLDER, "*.png"))
if len(images) == 0:
    raise Exception("No calibration images found in folder!")

print(f"Found {len(images)} images.")

for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if ret:
        objpoints.append(objp)

        # Improve corner accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners2, ret)
        cv2.imshow("Detected Corners", img)
        cv2.waitKey(200)
    else:
        print(f"❌ Chessboard not found in: {fname}")

cv2.destroyAllWindows()

# Calibration
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("\n✅ Calibration Done")
print("Camera matrix:\n", camera_matrix)
print("Distortion coefficients:\n", dist_coeffs)

# Save parameters
np.savez(SAVE_FILE,
         camera_matrix=camera_matrix,
         dist_coeffs=dist_coeffs)

print(f"\n✅ Saved calibration to: {SAVE_FILE}")
