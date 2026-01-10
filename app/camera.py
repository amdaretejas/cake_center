from matplotlib.pyplot import box
import cv2

class Camera:
    def __init__(self, camera_index=1):
        self.cam = cv2.VideoCapture(camera_index)
        print(f"Camera initialized at index {camera_index}")

    def stop(self):
        self.cam.release()
        cv2.destroyAllWindows()

    def add_middle_line(self, frame):
        height, width, _ = frame.shape
        cv2.line(frame, (width // 2, 0), (width // 2, height), (0, 255, 0), 1)
        cv2.line(frame, (0, height // 2), (width, height // 2), (0, 255, 0), 1)
        return frame
    
    def add_origin(self, frame):
        height, width, _ = frame.shape
        cv2.circle(frame, (0, 0), 12, (255, 0, 0), 2) # Origin point
        cv2.line(frame, (0, 0), (12, 0), (255, 255, 0), 2)
        cv2.line(frame, (0, 0), (0, 12), (255, 255, 0), 2)
        return frame

    def add_center(self, frame, result=[], offset_x=0, offset_y=0):
        if result == []:
            height, width, _ = frame.shape
            cv2.circle(frame, (width // 2, height // 2), 4, (255, 0, 0), 1)
        else:
            cv2.circle(frame, (int(result[0]) + offset_x, int(result[1]) + offset_y), 4, (0, 0, 255), 1)
        return frame

    def add_box(self, frame, result):
        if result == []:
            return frame
        cv2.rectangle(frame, 
                      (int(result[0]), int(result[1])), 
                      (int(result[2]), int(result[3])), 
                      (0, 0, 255), 1)
        return frame
    
    def add_box_obb(self, frame, result):
        if result == []:
            return frame
        print(result)
        cv2.line(frame, 
                 (int(result[0]), int(result[1])),
                 (int(result[2]), int(result[3])), 
                 (0, 0, 255), 1)
        cv2.line(frame, 
                 (int(result[2]), int(result[3])),
                 (int(result[4]), int(result[5])), 
                 (0, 0, 255), 1)
        cv2.line(frame, 
                 (int(result[4]), int(result[5])),
                (int(result[6]), int(result[7])), 
                (0, 0, 255), 1) 
        cv2.line(frame, 
                 (int(result[6]), int(result[7])),
                (int(result[0]), int(result[1])), 
                (0, 0, 255), 1)
        return frame

    def cut_frame(self, frame, result, cut_x, cut_y):
        if result == [] or cut_x <=0 or cut_y <=0 or len(result) !=4:
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

    def cut_frame_obb(self, frame, result, cut_x, cut_y):
        if result == [] or cut_x <=0 or cut_y <=0 or len(result) !=8:
            return frame
        # x1, y1, x2, y2 = map(int, result)
        # x_diff = (x2 - x1) // cut_x
        # y_diff = (y2 - y1) // cut_y
        new_x1, new_y1 = (result[0] + result[6]) / 2, (result[1] + result[7]) / 2
        new_x2, new_y2 = (result[2] + result[4]) / 2, (result[3] + result[5]) / 2
        new_x3, new_y3 = (result[0] + result[2]) / 2, (result[1] + result[3]) / 2
        new_x4, new_y4 = (result[6] + result[4]) / 2, (result[7] + result[5]) / 2

        for i in range(1, cut_x):

            cv2.line(frame, 
                        (int((new_x1 + new_x3)/2), int((new_y1 + new_y3)/2)), 
                        (int((new_x2 + new_x4)/2), int((new_y2 + new_y4)/2)), 
                        (255, 0, 0), 1) 

        # for j in range(1, cut_y):
        cv2.line(frame,
                    (int((result[0] + result[2])/2), int((result[1] + result[3])/2)), 
                    (int((result[6] + result[4])/2), int((result[7] + result[5])/2)), 
                    (255, 0, 0), 1)
        return frame


    def show_frame(self, frame, window_name="Camera"):
        cv2.imshow(window_name, frame)

    def wait_key(self, key='q'):
        return cv2.waitKey(1) & 0xFF == ord(key)
    
    def destroy_all_windows(self):
        cv2.destroyAllWindows()