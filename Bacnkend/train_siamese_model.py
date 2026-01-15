# =============================================================
# Siamese Network Training Script for Facial Recognition
# =============================================================

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import cv2
from sklearn.model_selection import train_test_split

# =============================================================
# CONFIGURATION
# =============================================================
DATASET_PATH = "dataset/"      # Folder with subfolders for each person
MODEL_SAVE_PATH = "model/siamese_model.h5"
IMG_SIZE = (100, 100)
EPOCHS = 20
BATCH_SIZE = 16

# =============================================================
# STEP 1: Load Images
# Each person should have a subfolder: dataset/person_name/*.jpg
# =============================================================

def load_images_from_folders(base_path):
    images = []
    labels = []
    persons = os.listdir(base_path)
    for label, person in enumerate(persons):
        person_folder = os.path.join(base_path, person)
        if not os.path.isdir(person_folder):
            continue
        for file in os.listdir(person_folder):
            if file.endswith(".jpg") or file.endswith(".png"):
                img = cv2.imread(os.path.join(person_folder, file))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, IMG_SIZE)
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels), persons

images, labels, class_names = load_images_from_folders(DATASET_PATH)
images = images.astype("float32") / 255.0

print(f"✅ Loaded {len(images)} images from {len(class_names)} persons.")

# =============================================================
# STEP 2: Create Positive and Negative Pairs
# =============================================================

def create_pairs(images, labels):
    pairs = []
    targets = []

    num_classes = len(np.unique(labels))
    idx = [np.where(labels == i)[0] for i in range(num_classes)]

    for class_idx in range(num_classes):
        same_class = idx[class_idx]
        for i in range(len(same_class) - 1):
            # Positive pair (same person)
            img1, img2 = images[same_class[i]], images[same_class[i + 1]]
            pairs.append([img1, img2])
            targets.append(1)

            # Negative pair (different person)
            neg_class = (class_idx + random.randint(1, num_classes - 1)) % num_classes
            img1, img2 = images[same_class[i]], images[idx[neg_class][0]]
            pairs.append([img1, img2])
            targets.append(0)

    return np.array(pairs), np.array(targets)

pairs, targets = create_pairs(images, labels)
print(f"✅ Created {len(pairs)} training pairs.")

train_pairs, test_pairs, train_y, test_y = train_test_split(pairs, targets, test_size=0.2, random_state=42)

# =============================================================
# STEP 3: Define Base CNN Network
# =============================================================

def build_base_cnn(input_shape):
    model = tf.keras.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='sigmoid')
    ])
    return model

# =============================================================
# STEP 4: Build Siamese Network
# =============================================================

def build_siamese_network(input_shape):
    base_cnn = build_base_cnn(input_shape)

    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)

    feats_a = base_cnn(input_a)
    feats_b = base_cnn(input_b)

    # L1 distance between feature vectors
    distance = layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]), output_shape=(128,))([feats_a, feats_b])
    output = layers.Dense(1, activation='sigmoid')(distance)

    siamese_net = Model(inputs=[input_a, input_b], outputs=output)
    return siamese_net

model = build_siamese_network(IMG_SIZE + (3,))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# =============================================================
# STEP 5: Prepare Data Generator
# =============================================================

def generate_batch(pairs, labels, batch_size):
    while True:
        idx = np.random.choice(len(pairs), batch_size)
        batch_pairs = pairs[idx]
        batch_labels = labels[idx]
        x1 = np.stack(batch_pairs[:, 0])
        x2 = np.stack(batch_pairs[:, 1])
        yield [x1, x2], batch_labels

# =============================================================
# STEP 6: Train the Model
# =============================================================

# Prepare the data for the model's two inputs
x_train_1 = train_pairs[:, 0]  # All the first images in the pairs
x_train_2 = train_pairs[:, 1]  # All the second images in the pairs

x_test_1 = test_pairs[:, 0]    # All the first images for validation
x_test_2 = test_pairs[:, 1]    # All the second images for validation

# Train the model directly with the NumPy arrays
history = model.fit(
    [x_train_1, x_train_2],
    train_y,
    validation_data=([x_test_1, x_test_2], test_y),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS
)
# =============================================================
# STEP 7: Save Model
# =============================================================

# Extract the base CNN model (the embedding generator)
base_cnn = model.get_layer('sequential')
EMBEDDING_MODEL_SAVE_PATH = "model/embedding_model.h5"
base_cnn.save(EMBEDDING_MODEL_SAVE_PATH)
print(f"✅ Embedding model saved successfully at {EMBEDDING_MODEL_SAVE_PATH}")

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
model.save(MODEL_SAVE_PATH)
print(f"✅ Siamese training model saved successfully at {MODEL_SAVE_PATH}")