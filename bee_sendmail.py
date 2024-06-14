import os
import sys
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
from dataclasses import dataclass

import smtplib
import ssl

# Assuming you cloned ByteTrack in your user directory
# Adjust the path to your ByteTrack directory
bytetrack_path = os.path.abspath("D:\\Pycharm Projects\\yolo_bee\\ByteTrack")
sys.path.append(bytetrack_path)

print("supervision.__version__:", sv.__version__)

VIDEOS_DIR = os.path.join('.', 'video_for_yolo')
SOURCE_VIDEO_PATH = "E:\\video_for_yolo\\bee_01.mp4"

# Model settings
MODEL = "runs/detect/train/weights/best.pt"
model = YOLO(MODEL)
model.fuse()

# Class settings
CLASS_NAMES_DICT = model.model.names
selected_classes = [0]

# Video settings
LINE_START = sv.Point(1000, 0)
LINE_END = sv.Point(1000, 2250)
TARGET_VIDEO_PATH = "E:\\video_for_yolo\\bee_01_counter_result.mp4"
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

# Tracking settings
byte_tracker = sv.ByteTrack(track_thresh=0.25, track_buffer=30, match_thresh=0.8, frame_rate=30)

# Line zone settings
line_zone = sv.LineZone(start=LINE_START, end=LINE_END)

# Annotators
box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
trace_annotator = sv.TraceAnnotator(thickness=4, trace_length=50)
line_zone_annotator = sv.LineZoneAnnotator(thickness=4, text_thickness=4, text_scale=2)

# Counters for bees in and out
bees_in = 0
bees_out = 0

def callback(frame: np.ndarray, index:int) -> np.ndarray:
    global bees_in, bees_out
    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[np.isin(detections.class_id, selected_classes)]
    detections = byte_tracker.update_with_detections(detections)
    labels = [
        f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
        for confidence, class_id, tracker_id
        in zip(detections.confidence, detections.class_id, detections.tracker_id)
    ]
    annotated_frame = trace_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )
    annotated_frame = box_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels
    )
    line_zone.trigger(detections)
    if line_zone.in_count > bees_in:
        bees_in += line_zone.in_count - bees_in
    if line_zone.out_count > bees_out:
        bees_out += line_zone.out_count - bees_out
    return line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)

sv.process_video(
    source_path=SOURCE_VIDEO_PATH,
    target_path=TARGET_VIDEO_PATH,
    callback=callback
)

# Email settings
smtp_port = 587
smtp_server = "smtp.gmail.com"
email_sender = "hfjulien2001@graduate.utm.my"
email_receiver = "hfjulien2001@graduate.utm.my"
pswd = "npyfzrwzrvmsyzgp"
message = f"Subject: Bee Count Report\n\nNumber of bees in: {bees_in}\nNumber of bees out: {bees_out}"

simple_email_context = ssl.create_default_context()

try:
    print("Connecting to server...")
    TIE_server = smtplib.SMTP(smtp_server, smtp_port)
    TIE_server.starttls(context=simple_email_context)
    TIE_server.login(email_sender, pswd)
    print("Connected to server!!")
    print(f"Sending email to {email_receiver}")
    TIE_server.sendmail(email_sender, email_receiver, message)
    print(f"Email successfully sent to {email_receiver}")
except Exception as e:
    print(e)
finally:
    TIE_server.quit()
