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
yolo_model = YoloModel(model_path="runs/obb/train10/weights/best.pt")
# yolo_model = YoloModel(model_path="yolov8n.pt")

running = True

def signal_handler(sig, frame):
    global running
    print("Ctrl+C detected. Shutting down safely...")
    running = False

signal.signal(signal.SIGINT, signal_handler)

REG_TRIGGER = 0
REG_LIVE = 1
REG_PROCESSING = 2
REG_X_MM = 4
REG_Y_MM = 3

x_center_offset = 0 #+5  # pixels
y_center_offset = 0 #-15  # pixels

x_offset = 0
y_offset = 0

pix_conversion = 0.817  # mm per pixel
pix_conversion_x = 0.817  # mm per pixel
pix_conversion_y = 0.817  # mm per pixel

x_mm = 0
y_mm = 0
x_actual = 0
y_actual = 0
last_listning_value = 0
current_listning_value = 0

try:
    while running:
        modbus_client.set_register(REG_LIVE, 1)
        ret, frame = camera.cam.read()
        if not ret:
            break

        current_listning_value = modbus_client.get_register(REG_TRIGGER)
        # print(f"Listening value at register {REG_TRIGGER}: {listening_value}")
        if current_listning_value == 1 and last_listning_value == 0:
            modbus_client.set_register(REG_PROCESSING, 1)
            results = yolo_model.predict(frame)
            x1,y1,x2,y2=0,0,0,0
            diay=0
            diax=0
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
                frame,x1,y1,x2,y2 = camera.add_box_obb(frame, object_box)
                diay = y2 - y1
                diax = x2 - x1

                if object_box != []:
                    frame = camera.cut_frame(frame, [object_box[0], object_box[1], object_box[4], object_box[5]], cut_x=5, cut_y=7)
                if object_center != []:
                    x_actual = int(object_center[0])
                    y_actual = int(object_center[1])
                    x_mm = int(object_center[0] * pix_conversion)
                    y_mm = int(object_center[1] * pix_conversion)
                modbus_client.set_register(REG_X_MM, x_mm)
                modbus_client.set_register(REG_Y_MM, y_mm)
                modbus_client.set_register(REG_PROCESSING, 0)
                print("--------------------------------")
                print(f"Listening value at register {REG_TRIGGER}: {modbus_client.get_register(REG_TRIGGER)}")
                print(f"Listening value at register {REG_PROCESSING}: {modbus_client.get_register(REG_PROCESSING)}")
                print(f"Listening value at register {REG_X_MM}: {modbus_client.get_register(REG_X_MM)} : {x_actual} pixels")
                print(f"Listening value at register {REG_Y_MM}: {modbus_client.get_register(REG_Y_MM)} : {y_actual} pixels")
                print(f"Listening value at register {REG_TRIGGER}: {modbus_client.get_register(REG_TRIGGER)}")
                print(f"Diameter in Y {diay}: {diay * pix_conversion_y} mm")
                print(f"Diameter in X {diax}: {diax * pix_conversion_x} mm")
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
