from pymodbus.client import ModbusTcpClient
import time


host = "192.168.1.20"
port = 502

register1 = 0
register2 = 1
register3 = 2
register4 = 3

client = ModbusTcpClient(host=host, port=port)
client.connect()

client.write_register(register4, 10)

while True:
    val = client.read_holding_registers(register1)
    print(f"Register {register1} value: {val.registers[0]}")
    val = client.read_holding_registers(register2)
    print(f"Register {register2} value: {val.registers[0]}")
    val = client.read_holding_registers(register3)
    print(f"Register {register3} value: {val.registers[0]}")
    val = client.read_holding_registers(register4)
    print(f"Register {register4} value: {val.registers[0]}")
    time.sleep(1)

    # val = client.read_holding_registers(register1, 1)
    # print(f"Register {register1} value: {val.registers[0]}")

    # client.write_register(register2, 456)
    # val = client.read_holding_registers(register2, 1)
    # print(f"Register {register2} value: {val.registers[0]}")

    # client.write_register(register3, 789)
    # val = client.read_holding_registers(register3, 1)
    # print(f"Register {register3} value: {val.registers[0]}")

    # client.write_register(register4, 1011)
    # val = client.read_holding_registers(register4, 1)
    # print(f"Register {register4} value: {val.registers[0]}")

    # client.close()