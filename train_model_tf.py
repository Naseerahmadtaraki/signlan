import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys
import io

# Set UTF-8 encoding for output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load the data with error handling
try:
    data_dict = pickle.load(open('./data.pickle', 'rb'))
except FileNotFoundError:
    raise FileNotFoundError("The file './data.pickle' was not found.")
except Exception as e:
    raise ValueError(f"An error occurred while loading the data: {e}")

# Extract data and labels
data = data_dict['data']
labels = np.asarray(data_dict['labels'])

# Validate the data format
if not all(isinstance(i, (list, np.ndarray)) and np.array(i).dtype.kind in 'biufc' for i in data):
    raise ValueError("Data must be a list or array of numerical sequences.")

# Pad sequences to the maximum length if necessary
max_length = max(len(x) for x in data)
padded_data = np.array([np.pad(x, (0, max_length - len(x)), 'constant') for x in data])

# Convert labels to integers
labels = labels.astype('int')

# Ensure labels are valid for stratification
if len(np.unique(labels)) < 2:
    raise ValueError("Stratification requires at least two classes in labels.")

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(
    padded_data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# Build a simple neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(padded_data.shape[1],)),  # Input layer
    tf.keras.layers.Dense(64, activation='relu'),          # Hidden layer
    tf.keras.layers.Dense(32, activation='relu'),          # Hidden layer
    tf.keras.layers.Dense(len(np.unique(labels)), activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'{accuracy * 100:.2f}% of samples were classified correctly!')

# Save the model in TensorFlow format
model.save('model_tf.h5')

# Convert the model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
