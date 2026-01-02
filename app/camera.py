from matplotlib.pyplot import box
from ultralytics import YOLO
import cv2

# cam = cv2.VideoCapture(1)

# def cut_parts(x1, y1, x2, y2, x_cut, y_cut):
#     cut_cordinates = []
#     x_diff = (x2 - x1) // x_cut
#     y_diff = (y2 - y1) // y_cut
#     for i in range(1, x_cut):
#         cut_cordinates.append([(x1 + i * x_diff, y1), (x1 + i * x_diff, y2)])

#     for j in range(1, y_cut):
#         cut_cordinates.append([(x1, y1 + j * y_diff), (x2, y1 + j * y_diff)])

#     return cut_cordinates

# model = YOLO("yolov8n.pt")
# while True:
#     ret, frame = cam.read()
#     if not ret:
#         break
#     results = model(frame)
#     for result in results: 
#         annotated_frame = result.plot()
#         cv2.imshow("YOLOv8 Detection", annotated_frame)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cam.release()
# cv2.destroyAllWindows()

class Camera:
    def __init__(self, camera_index=1, model_path="yolov8n.pt"):
        self.cam = cv2.VideoCapture(camera_index)

    def stop(self):
        self.cam.release()
        cv2.destroyAllWindows()

    def add_middle_line(self, frame):
        height, width, _ = frame.shape
        cv2.line(frame, (width // 2, 0), (width // 2, height), (0, 255, 0), 2)
        return frame
    
    def add_origin(self, frame):
        height, width, _ = frame.shape
        cv2.circle(frame, (0, 0), 12, (255, 0, 0), 2) # Origin point
        cv2.line(frame, (0, 0), (12, 0), (255, 255, 0), 2)
        cv2.line(frame, (0, 0), (0, 12), (255, 255, 0), 2)
        return frame

    def add_center(self, frame):
        height, width, _ = frame.shape
        cv2.circle(frame, (width // 2, height // 2), 4, (0, 0, 255), -1)
        return frame

    def add_box(self, frame, result):
        if result is None or len(result) < 4:
            return frame
        cv2.rectangle(frame, 
                      (int(result[0]), int(result[1])), 
                      (int(result[2]), int(result[3])), 
                      (0, 255, 0), 2)
        return frame
    
    def add_box_obb(self, frame, result):
        if result is None or len(result) < 8:
            return frame
        cv2.line(frame, 
                 (int(result[0]), int(result[1])),
                 (int(result[2]), int(result[3])), 
                 (0, 255, 0), 2)
        cv2.line(frame, 
                 (int(result[2]), int(result[3])),
                 (int(result[4]), int(result[5])), 
                 (0, 255, 0), 2)
        cv2.line(frame, 
                 (int(result[4]), int(result[5])),
                (int(result[6]), int(result[7])), 
                (0, 255, 0), 2) 
        cv2.line(frame, 
                 (int(result[6]), int(result[7])),
                (int(result[0]), int(result[1])), 
                (0, 255, 0), 2)
        return frame

    def cut_frame(self, frame, result, cut_x, cut_y):
        if result is None or len(result) < 4:
            return frame
        x1, y1, x2, y2 = map(int, result)
        x_diff = (x2 - x1) // cut_x
        y_diff = (y2 - y1) // cut_y
        for i in range(1, cut_x):
            cv2.line(frame, 
                     (x1 + i * x_diff, y1), 
                     (x1 + i * x_diff, y2), 
                     (255, 0, 0), 1)

        for j in range(1, cut_y):
            cv2.line(frame, 
                     (x1, y1 + j * y_diff), 
                     (x2, y1 + j * y_diff), 
                     (255, 0, 0), 1)
        return frame

    def show_frame(self, frame, window_name="Camera"):
        cv2.imshow(window_name, frame)

    def wait_key(self, key='q'):
        return cv2.waitKey(1) & 0xFF == ord(key)