import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Load your dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "your_dataset/train", 
    image_size=(32, 32), 
    batch_size=32
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "your_dataset/test", 
    image_size=(32, 32), 
    batch_size=32
)

# Get class names
class_names = train_ds.class_names
num_classes = len(class_names)

# Prefetching for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Build CNN model
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(32, 32, 3)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    
     layers.Conv2D(64, 3, activation='relu'),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
history = model.fit(train_ds, validation_data=test_ds, epochs=10)

# Evaluate model
test_loss, test_acc = model.evaluate(test_ds)
print(f"\nTest Accuracy: {test_acc:.2f}")

# Plot
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()