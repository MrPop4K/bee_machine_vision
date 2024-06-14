from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s.yaml")  # build a new model from scratch


# Use the model
model.train(data="config.yaml", epochs=100)  # train the model

#SKRIPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP

import os

from ultralytics import YOLO
import cv2

import sys
import yolox
print("yolox.__version__:", yolox.__version__)
import supervision
print("supervision.__version__:", supervision.__version__)

import supervision
import matplotlib.pyplot as plt
from supervision.draw.color import ColorPalette
from supervision.video.source import get_video_frames_generator
from supervision.tools.detections import Detections, BoxAnnotator

from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from supervision.notebook.utils import show_frame_in_notebook
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator

# Assuming you cloned ByteTrack in your user directory
# Adjust the path to your ByteTrack directory
bytetrack_path = os.path.abspath("D:\Pycharm Projects\yolo_bee\ByteTrack")
sys.path.append(bytetrack_path)

from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass

@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

VIDEOS_DIR = os.path.join('.', 'video_for_yolo')

video_path = os.path.join(VIDEOS_DIR, 'bee_01.mp4')
SOURCE_VIDEO_PATH = os.path.join(VIDEOS_DIR, 'bee_01.mp4')
video_path_out = '{}_out.mp4'.format(video_path)
# Your tracking code here

# settings
MODEL = "runs/detect/train/weights/best.pt"
model = YOLO(MODEL)
model.fuse()

#Predict and annotate single frame (Just for testing) Comment all of this out for video use
# dict maping class_id to class_name
CLASS_NAMES_DICT = model.model.names
# class_ids of interest - car, motorcycle, bus and truck
CLASS_ID = [0]

# create frame generator
generator = get_video_frames_generator(SOURCE_VIDEO_PATH)
# create instance of BoxAnnotator
box_annotator = BoxAnnotator(color=ColorPalette(), thickness=4, text_thickness=4, text_scale=2)
# acquire first video frame
iterator = iter(generator)
frame = next(iterator)
# model prediction on single frame and conversion to supervision Detections
results = model(frame)
detections = Detections(
    xyxy=results[0].boxes.xyxy.cpu().numpy(),
    confidence=results[0].boxes.conf.cpu().numpy(),
    class_id=results[0].boxes.cls.cpu().numpy().astype(int)
)
# format custom labels
labels = [
    f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
    for _, confidence, class_id, tracker_id
    in detections
]
# annotate and display frame
frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)

# Display frame using matplotlib
plt.figure(figsize=(16, 16))
plt.imshow(frame)
plt.axis('off')
plt.show()  #Comment up till here for single frame !!


