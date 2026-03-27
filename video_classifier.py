# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# # Load your trained model
# model = load_model('models/food_classifier.h5')

# # Define class labels (same as used during training)
# class_labels = {0: 'food', 1: 'not_food'}

# # Open a video file (or use 0 for webcam)
# video_path = '/Users/afatullahsiddique/Documents/food_video.mp4'  # replace with your video file path
# cap = cv2.VideoCapture(video_path)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Resize frame to model input size
#     img = cv2.resize(frame, (224, 224))
#     img_array = np.expand_dims(img, axis=0)
#     img_array = preprocess_input(img_array)

#     # Make prediction
#     pred = model.predict(img_array)
#     probability = pred[0][0]  # Get the single probability value
    
#     if probability > 0.5:
#         label = 'not_food'  # Class 1
#         confidence = probability
#     else:
#         label = 'food'      # Class 0
#         confidence = 1 - probability

#     # Put label on the frame
#     cv2.putText(frame, f"{label}: {confidence*100:.2f}%", (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#     # Show the frame
#     cv2.imshow('Video Classification', frame)

#     # Press 'q' to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
