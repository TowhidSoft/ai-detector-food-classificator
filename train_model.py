# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
# from tensorflow.keras.models import Model
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from tensorflow.keras.optimizers import Adam
# import os


# IMG_SIZE = (224, 224)
# BATCH_SIZE = 32
# EPOCHS = 5

# TRAIN_DIR = "dataset/train"
# TEST_DIR = "dataset/test"
# MODEL_PATH = "models/food_classifier.h5"


# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# train_datagen = ImageDataGenerator(
#     preprocessing_function=preprocess_input,
#     rotation_range=20,
#     zoom_range=0.2,
#     horizontal_flip=True
# )

# test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# train_data = train_datagen.flow_from_directory(
#     TRAIN_DIR,
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode="binary"
# )

# test_data = test_datagen.flow_from_directory(
#     TEST_DIR,
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode="binary",
#     shuffle=False
# )

# print("Class labels:", train_data.class_indices)

# base_model = MobileNetV2(
#     weights="imagenet",
#     include_top=False,
#     input_shape=(224, 224, 3)
# )

# base_model.trainable = True

# for layer in base_model.layers[:-30]:
#     layer.trainable = False



# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dropout(0.3)(x)
# output = Dense(1, activation="sigmoid")(x)

# model = Model(inputs=base_model.input, outputs=output)

# model.compile(
#     optimizer=Adam(learning_rate=1e-5),
#     loss="binary_crossentropy",
#     metrics=["accuracy"]
# )

# model.summary()

# os.makedirs("models", exist_ok=True)

# callbacks = [
#     EarlyStopping(patience=3, restore_best_weights=True),
#     ModelCheckpoint(MODEL_PATH, save_best_only=True)
# ]

# model.fit(
#     train_data,
#     validation_data=test_data,
#     epochs=5,
#     callbacks=callbacks
# )

# print(f"Model saved at {MODEL_PATH}")




