# -*- coding: utf-8 -*-
"""FYP EfficientNetB0.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1F53dzmm5gDmrTxlYueBSuQ51ce2VrMJG
"""

#import libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from google.colab import drive
import zipfile
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.applications import Xception
import cv2
from skimage import feature
from collections import Counter

drive.mount('/content/drive')

zip_train = zipfile.ZipFile('/content/drive/MyDrive/openforensics.zip', 'r')
zip_train.extractall('/tmp')
zip_train.close()

train_real = '/tmp/openforensics/Dataset/Train/Real'
train_fake = '/tmp/openforensics/Dataset/Train/Fake'

image_train_real = os.listdir(train_real)
image_train_fake = os.listdir(train_fake)

print("Number of real train images:", len(image_train_real))
#print("Sample real image file:", image_train_real[0])

print("Number of fake train images:", len(image_train_fake))
#print("Sample fake image file:", image_train_fake[0])

# Define the directory paths
real_dir = '/tmp/openforensics/Dataset/Train/Real'
fake_dir = '/tmp/openforensics/Dataset/Train/Fake'

# Helper function to validate and load the first image
def load_first_image_from_directory(directory):
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Error: Directory '{directory}' does not exist.")
    files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files:
        raise FileNotFoundError(f"Error: No image files found in directory '{directory}'.")
    return cv2.imread(os.path.join(directory, files[0]), cv2.IMREAD_GRAYSCALE)

# Load the first images from each directory
real_image = load_first_image_from_directory(real_dir)
fake_image = load_first_image_from_directory(fake_dir)

# Apply Gaussian filter to reduce noise
real_image_filtered = cv2.GaussianBlur(real_image, (5, 5), 0)
fake_image_filtered = cv2.GaussianBlur(fake_image, (5, 5), 0)

# Define LBP parameters
radius = 1  # Radius around the center pixel
n_points = 8 * radius  # Number of sampling points

# Apply LBP to both filtered images
real_lbp = feature.local_binary_pattern(real_image_filtered, n_points, radius, method="uniform")
fake_lbp = feature.local_binary_pattern(fake_image_filtered, n_points, radius, method="uniform")

# Display the original, filtered, and LBP-transformed images for both real and fake
plt.figure(figsize=(12, 9))

# Real image, filtered, and its LBP
plt.subplot(3, 2, 1)
plt.title("Real Image")
plt.imshow(real_image, cmap="gray")
plt.axis("off")

plt.subplot(3, 2, 3)
plt.title("Real Image (Filtered)")
plt.imshow(real_image_filtered, cmap="gray")
plt.axis("off")

plt.subplot(3, 2, 5)
plt.title("Real Image LBP")
plt.imshow(real_lbp, cmap="gray")
plt.axis("off")

# Fake image, filtered, and its LBP
plt.subplot(3, 2, 2)
plt.title("Fake Image")
plt.imshow(fake_image, cmap="gray")
plt.axis("off")

plt.subplot(3, 2, 4)
plt.title("Fake Image (Filtered)")
plt.imshow(fake_image_filtered, cmap="gray")
plt.axis("off")

plt.subplot(3, 2, 6)
plt.title("Fake Image LBP")
plt.imshow(fake_lbp, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()

# Define the directory path
data_dir = '/tmp/openforensics/Dataset/Train'

# Use ImageDataGenerator to load file paths and labels
datagen = tf.keras.preprocessing.image.ImageDataGenerator()
data_flow = datagen.flow_from_directory(
    data_dir,
    target_size=(256, 256),
    batch_size=16,
    class_mode="binary",  # Binary labels for "real" and "fake"
    shuffle=True
)

# Retrieve file paths and labels
file_paths = [data_flow.filepaths[i] for i in range(len(data_flow.filepaths))]
labels = data_flow.labels.tolist()

# Count occurrences of each label and get a balanced sample size
label_counts = Counter(labels)
min_count = min(label_counts.values())
num_samples_per_class = 3000 // len(label_counts)
num_samples_per_class = min(num_samples_per_class, min_count)

# Create lists to store stratified samples
stratified_file_paths = []
stratified_labels = []

# Select samples for each class
for label in label_counts.keys():
    indices = [i for i, l in enumerate(labels) if l == label]
    selected_indices = np.random.choice(indices, size=num_samples_per_class, replace=False)
    stratified_file_paths.extend([file_paths[i] for i in selected_indices])
    stratified_labels.extend([labels[i] for i in selected_indices])

# Print sample counts for each class
print("Number of 'real' images:", Counter(stratified_labels)[0])  # '0' represents 'real'
print("Number of 'fake' images:", Counter(stratified_labels)[1])  # '1' represents 'fake'

# Helper function to extract LBP features and create image-like output with Gaussian smoothing
def extract_lbp(image_path, radius=1, n_points=8, smoothing_ksize=(5, 5)):
    # Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian smoothing to reduce noise
    image_smoothed = cv2.GaussianBlur(image, smoothing_ksize, 0)

    # Apply LBP transformation
    lbp = feature.local_binary_pattern(image_smoothed, n_points * radius, radius, method="uniform")

    # Normalize LBP image to range [0, 255] and convert to uint8
    lbp_image = ((lbp - lbp.min()) / (lbp.max() - lbp.min()) * 255).astype(np.uint8)

    # Resize LBP image to the desired input size for the model (256x256)
    lbp_image_resized = cv2.resize(lbp_image, (256, 256))
    return lbp_image_resized

# Helper function to load LBP-transformed images and labels into a TensorFlow dataset
def paths_to_lbp_dataset(paths, labels, batch_size=32):
    lbp_images = []

    # Apply LBP transformation to each image
    for path in paths:
        lbp_image = extract_lbp(path)
        lbp_images.append(lbp_image)

    # Convert to TensorFlow dataset
    image_ds = tf.data.Dataset.from_tensor_slices(np.array(lbp_images).reshape(-1, 256, 256, 1))
    label_ds = tf.data.Dataset.from_tensor_slices(labels)

    # Combine images and labels into one dataset
    dataset = tf.data.Dataset.zip((image_ds, label_ds)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Create the TensorFlow dataset for the LBP-transformed images
stratified_dataset = paths_to_lbp_dataset(stratified_file_paths, stratified_labels)

# Print dataset size
print("Stratified dataset size:", len(stratified_file_paths))

# Shuffle stratified samples
indices = np.arange(len(stratified_file_paths))
np.random.shuffle(indices)  # Shuffle the indices

# Reorder file paths and labels based on shuffled indices
stratified_file_paths = [stratified_file_paths[i] for i in indices]
stratified_labels = [stratified_labels[i] for i in indices]

import pandas as pd  # Import pandas to handle DataFrame
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Specify the input shape explicitly as (256, 256, 3) for RGB images
efficientnet_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
efficientnet_model.trainable = False

# Freeze all layers except the last 20
for layer in efficientnet_model.layers[:-1]:
    layer.trainable = True

# Create a new model with a custom output layer
input_tensor = tf.keras.Input(shape=(256, 256, 3))
x = efficientnet_model(input_tensor)
x = layers.GlobalAveragePooling2D()(x)  # Reduces to a 1D vector, no need for Flatten()
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.5)(x)  # Dropout after Dense layer
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)  # Dropout after Dense layer
x = layers.Dropout(0.4)(x)  # Dropout after Dense layer
x = layers.Dropout(0.3)(x)  # Dropout after Dense layer
output_tensor = layers.Dense(1, activation='sigmoid')(x)
model = models.Model(inputs=input_tensor, outputs=output_tensor)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00003),
              loss='binary_crossentropy',  # Binary classification
              metrics=['accuracy'])

# Print a summary of the model architecture
model.summary()

# Define the image data generators with on-the-fly augmentation for training data only
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

valid_datagen = ImageDataGenerator(rescale=1./255)

# Create the TensorFlow datasets for train, validation, and test splits
# Assuming you have split the stratified dataset as follows
X_train, X_val, y_train, y_val = train_test_split(stratified_file_paths, stratified_labels, test_size=0.3, stratify=stratified_labels, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, stratify=y_val, random_state=42)

# Helper function to create a generator for training and validation datasets
def create_generator(file_paths, labels, datagen, batch_size=32):
    # Convert labels to string format
    labels_str = [str(label) for label in labels]

    return datagen.flow_from_dataframe(
        pd.DataFrame({'filename': file_paths, 'label': labels_str}),  # Use labels_str instead
        x_col='filename',
        y_col='label',
        target_size=(256, 256),
        class_mode='binary',
        batch_size=batch_size,
        shuffle=True
    )

# Create training and validation generators
train_data = create_generator(X_train, y_train, train_datagen, batch_size=32)
valid_data = create_generator(X_val, y_val, valid_datagen, batch_size=32)

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(train_data,
                    epochs=20,
                    validation_data=valid_data)

# Plot training & validation accuracy and loss
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(accuracy) + 1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, accuracy, label='Training accuracy')
plt.plot(epochs, val_accuracy, label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# Evaluate the model on test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_data = create_generator(X_test, y_test, test_datagen, batch_size=32)

test_loss, test_acc = model.evaluate(test_data)
print(f"Test accuracy: {test_acc * 100:.2f}%")

# Print training and validation results
final_training_acc = accuracy[-1]
final_training_loss = loss[-1]
final_val_acc = val_accuracy[-1]
final_val_loss = val_loss[-1]

print(f"Training Accuracy: {final_training_acc * 100:.2f}%")
print(f"Training Loss: {final_training_loss * 100:.2f}%")
print(f"Validation Accuracy: {final_val_acc * 100:.2f}%")
print(f"Validation Loss: {final_val_loss * 100:.2f}%")

# Print testing results
print(f"Testing Accuracy: {test_acc * 100:.2f}%")
print(f"Testing Loss: {test_loss * 100:.2f}%")

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Obtain predictions
# Predict probabilities for the test set
y_pred_probs = model.predict(test_data)  # Get probabilities
y_pred = (y_pred_probs > 0.5).astype("int32")  # Convert to binary labels

# Step 2: Flatten true and predicted labels (since test_data has batches)
y_test_flat = np.array(y_test).flatten()
y_pred_flat = y_pred.flatten()

# Step 3: Calculate metrics
# Confusion matrix
conf_matrix = confusion_matrix(y_test_flat, y_pred_flat)
print("Confusion Matrix:")
print(conf_matrix)

# Precision, Recall, F1 Score
precision = precision_score(y_test_flat, y_pred_flat)
recall = recall_score(y_test_flat, y_pred_flat)
f1 = f1_score(y_test_flat, y_pred_flat)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Save the model to an HDF5 file
#model.save('xception_model.h5')