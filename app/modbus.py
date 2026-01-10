from ultralytics import YOLO
import cv2
from pymodbus.server import StartTcpServer, ModbusSerialServer
from pymodbus.client import ModbusTcpClient
from pymodbus.datastore import ModbusSequentialDataBlock
from pymodbus.datastore import ModbusServerContext, ModbusSlaveContext
import threading
import time

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
        self.server_thread = threading.Thread(target=self._start_server, daemon=True)
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

class ModbusMaster:
    def __init__(self, host="127.0.0.1", port=8000):
        self.host = host
        self.port = port
        self.client = ModbusTcpClient(host=self.host, port=self.port)

        if self.client.connect():
            print(f"Connected to Modbus server at {self.host}:{self.port}")
        else:
            raise ConnectionError("Failed to connect to Modbus server")
        
    def set_register(self, register, value):
        self.client.write_register(register, value)

    def get_register(self, register):
        response = self.client.read_holding_registers(register)
        return response.registers[0]

    def close(self):
        self.client.close()
        print("Modbus client disconnected")

if __name__ == "__main__":
    # modbus_server = ModbusServer()
    # modbus_server.set_register(10, 1)
    # print(f"Register 10 set to {modbus_server.get_register(10)}")
    # time.sleep(10)
    # modbus_server.set_register(10, 0)
    pass