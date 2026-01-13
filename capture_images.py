import cv2
import os

# ---------------- SETTINGS ----------------
SAVE_DIR = "calib_imgs"     # folder to save calibration images
CAM_INDEX = 1               # 0 = default webcam, try 1 if external cam
# ------------------------------------------

# create folder if not exists
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(CAM_INDEX)

if not cap.isOpened():
    print("❌ Camera not opening! Check camera index.")
    exit()

img_count = 0

print("\n✅ Camera opened successfully")
print("Press 's' to SAVE image")
print("Press 'q' to QUIT\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame")
        break

    # show instructions on screen
    display = frame.copy()
    cv2.putText(display, "Press 's' to Save | Press 'q' to Quit",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Capture Calibration Images", display)

    key = cv2.waitKey(1) & 0xFF

    # Save image
    if key == ord('s'):
        img_count += 1
        filename = os.path.join(SAVE_DIR, f"calib_{img_count:03d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"✅ Saved: {filename}")

    # Quit
    elif key == ord('q'):
        print("👋 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
