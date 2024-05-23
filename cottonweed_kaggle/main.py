import os
import zipfile
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# Extract the dataset
local_zip = '/CottonWeedID15_Kaggle.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('cotton_weed_id')
zip_ref.close()

# Define directories
data_dir = 'cotton_weed_id/CottonWeedID15_500_UM/'

# Get list of class names (directories)
class_names = os.listdir(data_dir)
print("Classes found:", class_names)

# Define directories
data_dir = 'cotton_weed_id/CottonWeedID15_500_UM/'

# Get list of class names (directories)
class_names = os.listdir(data_dir)
print("Classes found:", class_names)

# Collect all image paths and their corresponding labels
image_paths = []
labels = []

for label, class_name in enumerate(class_names):
    class_dir = os.path.join(data_dir, class_name)
    for fname in os.listdir(class_dir):
        image_paths.append(os.path.join(class_dir, fname))
        labels.append(label)

# Split the dataset into training, validation, and test sets
train_paths, temp_paths, train_labels, temp_labels = train_test_split(image_paths, labels, test_size=0.2, random_state=42)
val_paths, test_paths, val_labels, test_labels = train_test_split(temp_paths, temp_labels, test_size=0.5, random_state=42)

# Define the preprocess_image function
def preprocess_image(file_path, label):
    image = tf.io.read_file(file_path)  # Read the image file
    image = tf.image.decode_jpeg(image, channels=3)  # Decode JPEG image to RGB format
    image = tf.image.resize(image, [224, 224])  # Resize to 224x224 pixels
    image = tf.cast(image, tf.float32) / 255.0  # Normalize pixel values to [0, 1]
    return image, label

# Function to load datasets
def load_dataset(file_paths, labels):
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))  # Create a dataset of file paths and labels
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)  # Apply preprocessing to each image
    return dataset

# Create TensorFlow datasets
train_ds = load_dataset(train_paths, train_labels)
val_ds = load_dataset(val_paths, val_labels)
test_ds = load_dataset(test_paths, test_labels)

# Add batching (computing efficiency), shuffling, and prefetching
BATCH_SIZE = 32

# Batching and Prefetching
train_ds = train_ds.shuffle(buffer_size=len(train_paths)).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)


# Define the custom callback
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') > 0.97:
            print("\nReached 97% accuracy so cancelling training!")
            self.model.stop_training = True


# Define a simple model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

# Generate model summary
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model with the callback
callbacks = myCallback()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=[callbacks]
)