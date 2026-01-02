from app.camera import Camera
from app.modbus import ModbusServer
from app.yolo import YoloModel


camera = Camera(camera_index=1)
modbus_server = ModbusServer(host="0.0.0.0", port=8000)
yolo_model = YoloModel(model_path="runs/obb/train6/weights/best.pt")

REG_LIVE = 0
REG_TRIGGER = 1
REG_X_MM = 2
REG_Y_MM = 3

pix_conversion = 0.817  # mm per pixel

x_mm = 0
y_mm = 0

while True:
    modbus_server.set_register(REG_LIVE, 1)
    ret, frame = camera.cam.read()
    if not ret:
        break

    listening_value = modbus_server.get_register(REG_TRIGGER)
    if listening_value == 1:
        results = yolo_model.predict(frame)
        for result in results:
            result
            center_obb = yolo_model.find_center_obb(result)
            box_obb = yolo_model.find_box_obb(result)
            frame = camera.add_middle_line(frame)
            frame = camera.add_origin(frame)
            frame = camera.add_center(frame)
            frame = camera.add_box_obb(frame, box_obb)
            if center_obb is not None:
                x_mm = int(center_obb[0] * pix_conversion)
                y_mm = int(center_obb[1] * pix_conversion)
            modbus_server.set_register(REG_X_MM, x_mm)
            modbus_server.set_register(REG_Y_MM, y_mm)
            modbus_server.set_register(REG_TRIGGER, 0)
            camera.show_frame(frame, window_name="YOLOv8 OBB Detection")
    
    camera.show_frame(frame, window_name="YOLOv8 OBB Detection")
    if camera.wait_key('q'):
        break