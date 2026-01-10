from app.camera import Camera
from app.modbus import ModbusServer, ModbusMaster
from app.yolo import YoloModel
import signal
import sys
import traceback
import time

camera = Camera(camera_index=1)
# modbus_server = ModbusServer(host="0.0.0.0", port=8000)
# modbus_client = ModbusMaster(host="192.168.1.20", port=502)
modbus_client = ModbusMaster(port=8000)
yolo_model = YoloModel(model_path="runs/obb/train9/weights/best.pt")
# yolo_model = YoloModel(model_path="yolov8n.pt")

running = True

def signal_handler(sig, frame):
    global running
    print("Ctrl+C detected. Shutting down safely...")
    running = False

signal.signal(signal.SIGINT, signal_handler)

REG_LIVE = 0
REG_TRIGGER = 1
REG_X_MM = 2
REG_Y_MM = 3

x_center_offset = +5  # pixels
y_center_offset = -15  # pixels

pix_conversion = 0.817  # mm per pixel

x_mm = 0
y_mm = 0

try:
    modbus_client.set_register(REG_LIVE, 0)
    modbus_client.set_register(REG_TRIGGER, 0)
    modbus_client.set_register(REG_X_MM, 0)
    modbus_client.set_register(REG_Y_MM, 0)
    while running:
        modbus_client.set_register(REG_LIVE, 1)
        ret, frame = camera.cam.read()
        if not ret:
            break

        listening_value = modbus_client.get_register(REG_TRIGGER)
        print(f"Listening value at register {REG_TRIGGER}: {listening_value}")
        if listening_value == 1:
            results = yolo_model.predict(frame)
            for result in results:
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
                if object_center != []:
                    x_mm = int(object_center[0] * pix_conversion)
                    y_mm = int(object_center[1] * pix_conversion)
                modbus_client.set_register(REG_X_MM, x_mm)
                modbus_client.set_register(REG_Y_MM, y_mm)
                print(f"Listening value at register {REG_X_MM}: {modbus_client.get_register(REG_X_MM)}")
                print(f"Listening value at register {REG_Y_MM}: {modbus_client.get_register(REG_Y_MM)}")
                print(f"Listening value at register {REG_TRIGGER}: {modbus_client.get_register(REG_TRIGGER)}")
                modbus_client.set_register(REG_TRIGGER, 0)
                camera.show_frame(frame, window_name="predicted frame")

        camera.show_frame(frame, window_name="Live frame")
        if not running:
            break
        if camera.wait_key('q'):
            break

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
