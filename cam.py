from ultralytics import YOLO
import cv2

cam = cv2.VideoCapture(1)

model = YOLO("yolov8n.pt")
while True:
    ret, frame = cam.read()
    if not ret:
        break
    results = model(frame)
    for result in results: 
        annotated_frame = result.plot()
        cv2.imshow("YOLOv8 Detection", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()