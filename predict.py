import os

from ultralytics import YOLO
import cv2

import sys
import yolox
print("yolox.__version__:", yolox.__version__)
import supervision
print("supervision.__version__:", supervision.__version__)

import supervision as sv
import numpy as np
import matplotlib.pyplot as plt



# Assuming you cloned ByteTrack in your user directory
# Adjust the path to your ByteTrack directory
bytetrack_path = os.path.abspath("D:\Pycharm Projects\yolo_bee\ByteTrack")
sys.path.append(bytetrack_path)



VIDEOS_DIR = os.path.join('.', 'video_for_yolo')

# video_path = os.path.join(VIDEOS_DIR, 'bee_01.mp4')
# SOURCE_VIDEO_PATH = os.path.join(VIDEOS_DIR, 'bee_01.mp4')
SOURCE_VIDEO_PATH = "E:\\video_for_yolo\\bee_01.mp4"
# video_path_out = '{}_out.mp4'.format(video_path)

# Your tracking code here
# settings
MODEL = "runs/detect/train/weights/best.pt"
model = YOLO(MODEL)
model.fuse()

# #Predict and annotate single frame (Just for testing) Comment all of this out for video use
# # dict maping class_id to class_name
# CLASS_NAMES_DICT = model.model.names
# # class_ids of interest - car, motorcycle, bus and truck
# CLASS_ID = [0]
#
# # create frame generator
# generator = get_video_frames_generator(SOURCE_VIDEO_PATH)
# # create instance of BoxAnnotator
# box_annotator = BoxAnnotator(color=ColorPalette(), thickness=4, text_thickness=4, text_scale=2)
# # acquire first video frame
# iterator = iter(generator)
# frame = next(iterator)
# # model prediction on single frame and conversion to supervision Detections
# results = model(frame)
# detections = Detections(
#     xyxy=results[0].boxes.xyxy.cpu().numpy(),
#     confidence=results[0].boxes.conf.cpu().numpy(),
#     class_id=results[0].boxes.cls.cpu().numpy().astype(int)
# )
# # format custom labels
# labels = [
#     f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
#     for _, confidence, class_id, tracker_id
#     in detections
# ]
# # annotate and display frame
# frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
# # Display frame using matplotlib
# plt.figure(figsize=(16, 16))
# plt.imshow(frame)
# plt.axis('off')
# plt.show()  #Comment up till here for single frame !!

#----------------------------------------------------------------------------------------------------------------

# #Predict and annotate single frame
# #dict maping class_id to class_name
# CLASS_NAMES_DICT = model.model.names
#
# # class_ids of interest - car, motorcycle, bus and truck
# selected_classes = [0]
# #
# # create frame generator
# generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
# # # create instance of BoxAnnotator
# box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
# # # acquire first video frame
# iterator = iter(generator)
# frame = next(iterator)
# # model prediction on single frame and conversion to supervision Detections
# results = model(frame, verbose=False)[0]
# #
# # convert to Detections
# detections = sv.Detections.from_ultralytics(results)
# # only consider class id from selected_classes define above
# detections = detections[np.isin(detections.class_id, selected_classes)]
# #
# # format custom labels
# labels = [
#     f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
#     for confidence, class_id in zip(detections.confidence, detections.class_id)
# ]
# #
# # annotate and display frame
# anotated_frame=box_annotator.annotate(scene=frame, detections=detections, labels=labels)
# #
# # Display frame using matplotlib
# plt.figure(figsize=(16, 16))
# plt.imshow(anotated_frame)
# plt.xlabel('X Axis')
# plt.ylabel('Y Axis')
# plt.grid(True)
# plt.show()
# ----------------------------------------------------------------------------------------------------------------
#Predict and annotate whole video
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