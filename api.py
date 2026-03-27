# import cv2
# import numpy as np
# import tempfile
# import os
# from fastapi import FastAPI, UploadFile, File
# from tensorflow.keras.models import load_model
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# app = FastAPI()

# model = load_model("models/food_classifier.h5")

# class_labels = {0: "food", 1: "not_food"}

# IMG_SIZE = 224
# THRESHOLD = 0.5


# def predict_frame(frame):
#     img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
#     img_array = np.expand_dims(img, axis=0)
#     img_array = preprocess_input(img_array)

#     pred = model.predict(img_array, verbose=0)
#     probability = pred[0][0]

#     if probability > THRESHOLD:
#         return "not_food", float(probability)
#     else:
#         return "food", float(1 - probability)

    
# @app.post("/predict-video")
# async def predict_video(file: UploadFile = File(...)):
#     # Save uploaded video temporarily
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
#         temp_video.write(await file.read())
#         video_path = temp_video.name

#     cap = cv2.VideoCapture(video_path)

#     food_frames = 0
#     not_food_frames = 0
#     total_frames = 0

#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     frame_interval = fps  # 1 frame per second

#     frame_id = 0
#     last_confidence = 0.0

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         if frame_id % frame_interval == 0:
#             label, confidence = predict_frame(frame)
#             last_confidence = confidence

#             if label == "food":
#                 food_frames += 1
#             else:
#                 not_food_frames += 1

#             total_frames += 1

#         frame_id += 1

#     cap.release()
#     os.remove(video_path)

#     final_label = "food" if food_frames > not_food_frames else "not_food"

#     return {
#         "result": final_label,
#         "confidence": round(last_confidence * 100, 2),
#         "food_frames": food_frames,
#         "not_food_frames": not_food_frames,
#         "checked_frames": total_frames
#     }
