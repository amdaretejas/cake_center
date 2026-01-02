from ultralytics import YOLO
import cv2
from pymodbus.server import StartTcpServer
from pymodbus.datastore import ModbusSequentialDataBlock
from pymodbus.datastore import ModbusServerContext, ModbusSlaveContext
import threading
import time

# host = "0.0.0.0"    
# port = 8000
# register1 = 10  # Address of the register to read

# store = ModbusSlaveContext(
#     di=ModbusSequentialDataBlock(0, [0]*100), # Discrete Input
#     co=ModbusSequentialDataBlock(0, [0]*100), # Coils
#     hr=ModbusSequentialDataBlock(0, [0]*100), # Holding Registers
#     ir=ModbusSequentialDataBlock(0, [0]*100), # Input Registers
# )

# context = ModbusServerContext(slaves=store, single=True)

# def _start_server():
#     print(f"Modbus TCP Server started on {host}:{port}")
#     StartTcpServer(context, address=(host, port))

# server_thread = threading.Thread(target=_start_server)
# server_thread.start()

# cam = cv2.VideoCapture(1)

# model = YOLO("yolov8n.pt")
# while True:
#     ret, frame = cam.read()
#     if not ret:
#         break
#     listning_value = store.getValues(3, register1, 1)[0]
#     if listning_value == 1:
#         results = model(frame)
#         for result in results: 
#             annotated_frame = result.plot()
#             cv2.imshow("YOLOv8 Detection", annotated_frame)
#             store.setValues(3, register1, [0])
#             time.sleep(2)
#     else:
#         cv2.imshow("YOLOv8 Detection", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cam.release()
# cv2.destroyAllWindows()

class ModbusServer:
    def __init__(self, host="0.0.0.0", port=8000):
        self.host = host
        self.port = port
        self.store = ModbusSlaveContext(
            di=ModbusSequentialDataBlock(0, [0]*100),
            co=ModbusSequentialDataBlock(0, [0]*100),
            hr=ModbusSequentialDataBlock(0, [0]*100),
            ir=ModbusSequentialDataBlock(0, [0]*100),
        )
        self.context = ModbusServerContext(slaves=self.store, single=True)
        self.server_thread = threading.Thread(target=self._start_server)
        self.server_thread.start()


    def _start_server(self):
        try:
            print(f"Modbus TCP Server started on {self.host}:{self.port}")
            StartTcpServer(self.context, address=(self.host, self.port))
        except Exception as e:
            print(f"Modbus server stopped: {e}")

    def set_register(self, register, value):
        self.store.setValues(3, register, [value])

    def get_register(self, register):
        return self.store.getValues(3, register, 1)[0]

if __name__ == "__main__":
    # modbus_server = ModbusServer()
    # modbus_server.set_register(10, 1)
    # print(f"Register 10 set to {modbus_server.get_register(10)}")
    # time.sleep(10)
    # modbus_server.set_register(10, 0)
    pass