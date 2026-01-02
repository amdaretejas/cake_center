from ultralytics import YOLO
import cv2
from pymodbus.server import StartTcpServer
from pymodbus.datastore import (
    ModbusSequentialDataBlock,
    ModbusServerContext,
    ModbusSlaveContext
)
import threading
import time
import signal
import sys

# -------------------------------
# Global running flag
# -------------------------------
running = True

def signal_handler(sig, frame):
    global running
    print("\n🛑 Ctrl+C detected. Shutting down safely...")
    running = False

signal.signal(signal.SIGINT, signal_handler)

# -------------------------------
# Modbus Configuration
# -------------------------------
HOST = "0.0.0.0"
PORT = 8000
pix_conversion = 0.817  # mm per pixel


REG_LIVE = 9
REG_TRIGGER = 10
REG_X = 11
REG_Y = 12
REG_H = 13
REG_W = 14
REG_CONF = 15
REG_X1 = 16
REG_Y1 = 17
REG_X2 = 18
REG_Y2 = 19
REG_X3 = 20
REG_Y3 = 21
REG_X4 = 22
REG_Y4 = 23

# -------------------------------
# Modbus Data Store
# -------------------------------
store = ModbusSlaveContext(
    di=ModbusSequentialDataBlock(0, [0]*100),
    co=ModbusSequentialDataBlock(0, [0]*100),
    hr=ModbusSequentialDataBlock(0, [0]*100),
    ir=ModbusSequentialDataBlock(0, [0]*100),
)

context = ModbusServerContext(slaves=store, single=True)

def cut_parts(x1, y1, x2, y2, x_cut, y_cut):
    cut_cordinates = []
    x_diff = (x2 - x1) // x_cut
    y_diff = (y2 - y1) // y_cut
    for i in range(1, x_cut):
        cut_cordinates.append([(x1 + i * x_diff, y1), (x1 + i * x_diff, y2)])

    for j in range(1, y_cut):
        cut_cordinates.append([(x1, y1 + j * y_diff), (x2, y1 + j * y_diff)])

    return cut_cordinates

def start_modbus():
    print(f"✅ Modbus TCP Server started on {HOST}:{PORT}")
    try:
        StartTcpServer(context, address=(HOST, PORT))
    except Exception as e:
        print(f"Modbus server stopped: {e}")

modbus_thread = threading.Thread(target=start_modbus, daemon=True)
modbus_thread.start()

# -------------------------------
# Camera & YOLO
# -------------------------------
cap = cv2.VideoCapture(1)
model = YOLO("yolov8n.pt")

print("🚀 System started. Press Ctrl+C to exit safely.")

# -------------------------------
# Main Loop
# -------------------------------
try:
    while running:
        store.setValues(3, REG_LIVE, [1])

        ret, frame = cap.read()
        if not ret:
            print("❌ Camera frame not received")
            break

        trigger = store.getValues(3, REG_TRIGGER, 1)[0]

        if trigger == 1:
            results = model(frame, verbose=False)

            if len(results[0].boxes) > 0:
                box = results[0].boxes[0]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                w = x2 - x1
                h = y2 - y1
                cx = x1 + w // 2
                cy = y1 + h // 2
                conf = int(box.conf[0] * 100)

                x3, y3 = x2, y1
                x4, y4 = x1, y2
                x_mm = int((cx * pix_conversion))
                y_mm = int((cy * pix_conversion))
                store.setValues(3, REG_X, [cx])
                store.setValues(3, REG_Y, [cy])
                store.setValues(3, REG_H, [x_mm])
                store.setValues(3, REG_W, [y_mm])
                # store.setValues(3, REG_W, [w])
                # store.setValues(3, REG_H, [h])
                # store.setValues(3, REG_CONF, [conf])
                # store.setValues(3, REG_X1, [x1])
                # store.setValues(3, REG_Y1, [y1])
                # store.setValues(3, REG_X2, [x2])
                # store.setValues(3, REG_Y2, [y2])
                # store.setValues(3, REG_X3, [x3])
                # store.setValues(3, REG_Y3, [y3])
                # store.setValues(3, REG_X4, [x4])
                # store.setValues(3, REG_Y4, [y4])
                updated_frame = frame.copy()
                cv2.rectangle(updated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(updated_frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.circle(updated_frame, (0, 0), 12, (0, 0, 255), 2) # Origin point
                cv2.line(updated_frame, (0, 0), (12, 0), (255, 255, 0), 2)
                cv2.line(updated_frame, (0, 0), (0, 12), (255, 255, 0), 2)
                cv2.line(updated_frame, (int(frame.shape[1]/2), 0), (int(frame.shape[1]/2), frame.shape[0]), (0, 0, 255), 1)
                cv2.line(updated_frame, (0, int(frame.shape[0]/2)), (frame.shape[1], int(frame.shape[0]/2)), (0, 0, 255), 1)
                cuts = cut_parts(x1, y1, x2, y2, 4, 4)
                for (start, end) in cuts:
                    cv2.line(updated_frame, start, end, (255, 0, 0), 1)
                cv2.imshow("YOLOv8 Detection", updated_frame)
                print(f"✅ Detected @ ({cx},{cy}) Conf:{conf/100:.2f}")
                print(f"   Box: x1={x1}, y1={y1}, x2={x2}, y2={y2}, w={w}, h={h}")

            else:
                print("⚠️ No object detected")

            store.setValues(3, REG_TRIGGER, [0])
            time.sleep(0.1)

        cv2.line(frame, (int(frame.shape[1]/2), 0), (int(frame.shape[1]/2), frame.shape[0]), (0, 0, 255), 1)
        cv2.line(frame, (0, int(frame.shape[0]/2)), (frame.shape[1], int(frame.shape[0]/2)), (0, 0, 255), 1)
        cv2.circle(frame, (0, 0), 12, (0, 0, 255), 2) # Origin point
        cv2.line(frame, (0, 0), (12, 0), (255, 255, 0), 2)
        cv2.line(frame, (0, 0), (0, 12), (255, 255, 0), 2)
        cv2.imshow("YOLOv8 live", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"🔥 Runtime error: {e}")

# -------------------------------
# Cleanup (VERY IMPORTANT)
# -------------------------------
print("🧹 Cleaning up resources...")

store.setValues(3, REG_LIVE, [0])
cap.release()
cv2.destroyAllWindows()

print("✅ Shutdown complete.")
sys.exit(0)
