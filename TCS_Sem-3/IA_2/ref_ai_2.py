import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Define data paths
data_dir = 'dataset/train'
classes = ['Covid', 'Normal', 'Viral Fever']
num_classes = len(classes)

# Image dimensions and batch size
img_width, img_height = 224, 224
batch_size = 32

# Load images and labels
data = []
labels = []

for class_index, class_name in enumerate(classes):
    class_dir = os.path.join(data_dir, class_name)
    for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_name)
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_width, img_height))
        image = tf.keras.preprocessing.image.img_to_array(image)
        data.append(image)
        labels.append(class_index)

# Convert lists to arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# Convert labels to one-hot encoded format
labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)

# Split the data into training and validation sets
train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

# Data augmentation for training images
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow(train_data, train_labels, batch_size=batch_size)
val_generator = val_datagen.flow(val_data, val_labels, batch_size=batch_size)

# Load ResNet50 with pretrained weights
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Add Global Average Pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Add a fully connected layer
x = Dense(512, activation='relu')(x)

# Add output layer for classification
output = Dense(num_classes, activation='softmax')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_data) // batch_size,
    epochs=30,
    validation_data=val_generator,
    validation_steps=len(val_data) // batch_size,
    callbacks=[early_stopping]
)

# Evaluate the model on the test set
test_data_dir = 'dataset/test'
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)
test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))
print("Test accuracy:", test_acc)
