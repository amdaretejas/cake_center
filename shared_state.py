import threading

lock = threading.Lock()

state = {
    "live_frame": None,
    "pred_frame": None,
    "readings": {},
    "inputs": {
        "x_offset": 0,
        "y_offset": 0,
        "conf": 0.5,
    },
    "logs": []
}
