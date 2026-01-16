from app.camera import Camera
from app.modbus import ModbusServer, ModbusMaster
from app.yolo import YoloModel
from shared_state import state, lock
import signal
import sys
import traceback
import time
import numpy as np
import cv2

data = np.load("camera_calibration.npz")
camera_matrix = data["camera_matrix"]
dist_coeffs = data["dist_coeffs"]

camera = Camera(camera_index=0)
modbus_client = ModbusMaster(host="192.168.1.20", port=502)
# modbus_client = ModbusMaster(port=8000)
yolo_model = YoloModel(model_path="runs/obb/train10/weights/best.pt")
# yolo_model = YoloModel(model_path="yolov8n.pt")

running = True

def signal_handler(sig, frame):
    global running
    print("Ctrl+C detected. Shutting down safely...")
    running = False

signal.signal(signal.SIGINT, signal_handler)

# REG_TRIGGER = 0 # this register is set to 1 to predict the object 
# REG_LIVE = 1 # this register shows if the system is live or not
# REG_PROCESSING = 2 # this register is set to 1 when processing is going on
# REG_X_MM = 4 # this register holds the X coordinate in mm
# REG_Y_MM = 3 # this register holds the Y coordinate in mm
# REG_STATUS = 5 # this register holds the status of the system

# write registers
REG_LIVE = 0
REG_PROCESSING = 1
REG_X_MM = 2
REG_Y_MM = 3
REG_STATUS = 4

# read registers
REG_TRIGGER = 5
REG_X_OFFSET = 6
REG_Y_OFFSET = 7
REG_PIX_TO_MM_X = 8
REG_PIX_TO_MM_Y = 9
REG_X1_LIMIT = 10
REG_Y1_LIMIT = 11
REG_X2_LIMIT = 12
REG_Y2_LIMIT = 13

x_center_offset = 0 #+5  # pixels
y_center_offset = 0 #-15  # pixels

x_px = 0 # pixels value in x axis
y_px = 0 # pixels value in y axis
h_px = 0 # pixels value in height
w_px = 0 # pixels value in width
x_mm = 0 # mm value in x axis
y_mm = 0 # mm value in y axis
h_mm = 0 # mm value in height
w_mm = 0 # mm value in width
x_machine_mm = 0 # machine coordinate in mm for x axis
y_machine_mm = 0 # machine coordinate in mm for y axis

x_hexadecimal = 0
y_hexadecimal = 0


last_listning_value = 0
current_listning_value = 0

prediction_status = 0 # 0 = no object, 1 = object detected, 2 = object out of bounds

try:
    modbus_client.set_register(REG_LIVE, 0)
    modbus_client.set_register(REG_PROCESSING, 0)
    modbus_client.set_register(REG_X_MM, 0)
    modbus_client.set_register(REG_Y_MM, 0)
    modbus_client.set_register(REG_STATUS, 0)
    modbus_client.set_register(REG_TRIGGER, 0)
    while running:
        pix_conversion_x = modbus_client.get_register(REG_PIX_TO_MM_X)/1000 # 1.84  # ratio of pixels to mm in x axis
        pix_conversion_y = modbus_client.get_register(REG_PIX_TO_MM_Y)/1000 # 1.84  # ratio of pixels to mm in y axis

        x_offset = modbus_client.get_register(REG_X_OFFSET) #281 # machine and real world origin offset in mm for x axis
        y_offset = modbus_client.get_register(REG_Y_OFFSET) #111 # machine and real world origin offset in mm for y axis

        x1_limit = modbus_client.get_register(REG_X1_LIMIT) # minimum x axis limits for cutting in mm
        y1_limit = modbus_client.get_register(REG_Y1_LIMIT) # minimum y axis limits for cutting in mm
        x2_limit = modbus_client.get_register(REG_X2_LIMIT) # maximum x axis limits for cutting in mm
        y2_limit = modbus_client.get_register(REG_Y2_LIMIT) # maximum y axis limits for cutting in mm
        modbus_client.set_register(REG_LIVE, 1)
        ret, frame = camera.cam.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 1, (w, h)
        )

        undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_mtx)
        x, y, w_roi, h_roi = roi
        undistorted = undistorted[y:y+h_roi, x:x+w_roi]
        frame = undistorted

        current_listning_value = modbus_client.get_register(REG_TRIGGER)
        # print(f"Listening value at register {REG_TRIGGER}: {listening_value}")
        if current_listning_value == 1 and last_listning_value == 0:
            modbus_client.set_register(REG_PROCESSING, 1)
            results = yolo_model.predict(frame)
            for result in results:
                names = [result.names[cls.item()] for cls in result.obb.cls.int()]  # class name of each box
                print("Detected object classes:", names)
                object_center = yolo_model.find_center_obb(result)
                object_box = yolo_model.find_box_obb(result)
                # object_center = yolo_model.find_center(result)
                # object_box = yolo_model.find_box(result) 
                frame = camera.add_middle_line(frame)
                frame = camera.add_origin(frame)
                frame = camera.add_center(frame)
                frame = camera.add_center(frame, object_center, offset_x=x_center_offset, offset_y=y_center_offset)
                frame = camera.add_box_obb(frame, object_box)
                if object_box != []:
                    frame = camera.cut_frame(frame, [object_box[0], object_box[1], object_box[4], object_box[5]], cut_x=5, cut_y=7)
                prediction_status = 0  # reset status
                if object_center != []:
                    object_center[0] = object_center[0] - x_center_offset
                    object_center[1] = object_center[1] - y_center_offset
                    x_px = int(object_center[0])
                    # y_px = int(object_center[1])
                    y_px = int(h - object_center[1])  # invert y axis
                    h_px = int(object_center[3])
                    w_px = int(object_center[2])
                    x_mm = int(x_px * pix_conversion_x)
                    y_mm = int(y_px * pix_conversion_y)
                    h_mm = int(h_px * pix_conversion_y)
                    w_mm = int(w_px * pix_conversion_x)
                    x_hexadecimal = hex(x_mm)
                    y_hexadecimal = hex(y_mm)
                    x_machine_mm = x_mm - x_offset
                    y_machine_mm = y_mm - y_offset
                    prediction_status = 1  # object detected successfully
                    if (x1_limit > x_machine_mm) or (x_machine_mm > x2_limit) or (y1_limit > y_machine_mm) or (y_machine_mm > y2_limit):
                        # x_machine_mm = 0
                        # y_machine_mm = 0
                        prediction_status = 2  # object out of bounds
                modbus_client.set_register(REG_X_MM, x_machine_mm if x_machine_mm >= 0 else 0)
                modbus_client.set_register(REG_Y_MM, y_machine_mm if y_machine_mm >= 0 else 0)
                modbus_client.set_register(REG_STATUS, prediction_status)
                modbus_client.set_register(REG_PROCESSING, 0)
                print("#/--------------------------------")
                print(f"Listening value at register {REG_PROCESSING}: {modbus_client.get_register(REG_PROCESSING)}")
                print(f"Listening value at register {REG_X_MM}: {modbus_client.get_register(REG_X_MM)}")
                print(f"Listening value at register {REG_Y_MM}: {modbus_client.get_register(REG_Y_MM)}")
                print(f"Listening value at register {REG_TRIGGER}: {modbus_client.get_register(REG_TRIGGER)}")
                print(f"Listening value at register {REG_STATUS}: {modbus_client.get_register(REG_STATUS)}")
                print(f"Listening value at register {REG_TRIGGER}: {modbus_client.get_register(REG_TRIGGER)}")
                print(f"Listening value at register {REG_X_OFFSET}: {modbus_client.get_register(REG_X_OFFSET)}")
                print(f"Listening value at register {REG_Y_OFFSET}: {modbus_client.get_register(REG_Y_OFFSET)}")
                print(f"Listening value at register {REG_PIX_TO_MM_X}: {modbus_client.get_register(REG_PIX_TO_MM_X)}")
                print(f"Listening value at register {REG_PIX_TO_MM_Y}: {modbus_client.get_register(REG_PIX_TO_MM_Y)}")
                print(f"Listening value at register {REG_X1_LIMIT}: {modbus_client.get_register(REG_X1_LIMIT)}")
                print(f"Listening value at register {REG_Y1_LIMIT}: {modbus_client.get_register(REG_Y1_LIMIT)}")
                print(f"Listening value at register {REG_X2_LIMIT}: {modbus_client.get_register(REG_X2_LIMIT)}")
                print(f"Listening value at register {REG_Y2_LIMIT}: {modbus_client.get_register(REG_Y2_LIMIT)}")
                print(f"#/--------------------------------")
                print("Detected object data:-")
                print(f"  Center X in mm: {x_mm} (px: {x_px})")
                print(f"  Center Y in mm: {y_mm} (px: {y_px})")
                print(f"  Center X machine mm: {x_machine_mm}")
                print(f"  Center Y machine mm: {y_machine_mm}")
                print(f"  X Hexadecimal: {x_hexadecimal}")
                print(f"  Y Hexadecimal: {y_hexadecimal}")
                print(f"  Height in mm: {h_mm} (px: {h_px})")
                print(f"  Width in mm: {w_mm} (px: {w_px})")
                # modbus_client.set_register(REG_TRIGGER, 0)
                camera.show_frame(frame, window_name="predicted frame")

        camera.show_frame(frame, window_name="Live frame")
        if not running:
            break
        if camera.wait_key('q'):
            break
        last_listning_value = current_listning_value
except Exception as e:
    print(f"Error occurred: {e}")
    traceback.print_exc()

finally:
    print("Shutting down system...")

    modbus_client.set_register(REG_LIVE, 0)

    camera.destroy_all_windows()
    camera.stop()
    
    print("Shutdown complete.")
    sys.exit(0)
