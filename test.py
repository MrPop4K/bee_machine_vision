import yolox
print("yolox.__version__:", yolox.__version__)

#Predict and annotate single frame
# dict maping class_id to class_name
CLASS_NAMES_DICT = model.model.names

# class_ids of interest - car, motorcycle, bus and truck
selected_classes = [0]
#
 # create frame generator
 generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
# # create instance of BoxAnnotator
 box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
# # acquire first video frame
 iterator = iter(generator)
 frame = next(iterator)
 # model prediction on single frame and conversion to supervision Detections
 results = model(frame, verbose=False)[0]
#
# # convert to Detections
 detections = sv.Detections.from_ultralytics(results)
# # only consider class id from selected_classes define above
 detections = detections[np.isin(detections.class_id, selected_classes)]
#
# # format custom labels
 labels = [
     f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
     for confidence, class_id in zip(detections.confidence, detections.class_id)
 ]
#
 # annotate and display frame
 anotated_frame=box_annotator.annotate(scene=frame, detections=detections, labels=labels)

# # Display frame using matplotlib
 plt.figure(figsize=(16, 16))
 plt.imshow(frame)
 plt.axis('off')
 plt.show()
#----------------------------------------------------------------------------------------------------------------