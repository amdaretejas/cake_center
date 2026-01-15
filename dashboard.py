import streamlit as st
import threading
import time
from core.shared_state import state, lock
from core.yolo_worker import start_pipeline

# ✅ Start detection only once (important)
if "pipeline_started" not in st.session_state:
    st.session_state.pipeline_started = False

def run_pipeline():
    start_pipeline(camera_index=0, conf=0.5)

st.set_page_config(layout="wide")
st.title("YOLO + MODBUS DASHBOARD")

# ✅ automatically start pipeline when dashboard starts
if not st.session_state.pipeline_started:
    t = threading.Thread(target=run_pipeline, daemon=True)
    t.start()
    st.session_state.pipeline_started = True
    st.success("✅ Detection Pipeline Started Automatically")

# --- placeholders for frames
col1, col2 = st.columns(2)
live_ph = col1.empty()
pred_ph = col2.empty()

# --- UI refresh loop
while True:
    with lock:
        live = state["live_frame"]
        pred = state["pred_frame"]
        fps = state["fps"]
        logs = state["logs"][-15:]

    col1.subheader("Live")
    col2.subheader("Prediction")

    if live is not None:
        live_ph.image(live, channels="BGR", use_container_width=True)

    if pred is not None:
        pred_ph.image(pred, channels="BGR", use_container_width=True)

    st.write("FPS:", fps)

    st.subheader("Logs")
    for l in logs[::-1]:
        st.write(l)

    time.sleep(0.05)
