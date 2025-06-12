import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 7-segment display truth table (CA configuration) [3][5]
SEGMENT_MAP = {
    0: [1,1,1,1,1,1,0],
    1: [0,1,1,0,0,0,0],
    2: [1,1,0,1,1,0,1],
    3: [1,1,1,1,0,0,1],
    4: [0,1,1,0,0,1,1],
    5: [1,0,1,1,0,1,1],
    6: [1,0,1,1,1,1,1],
    7: [1,1,1,0,0,0,0],
    8: [1,1,1,1,1,1,1],
    9: [1,1,1,1,0,1,1]
}

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Create 7-segment targets
y_train_seg = np.array([SEGMENT_MAP[label] for label in y_train])
y_test_seg = np.array([SEGMENT_MAP[label] for label in y_test])

# Convolutional Autoencoder [1][9]
latent_dim = 32
input_img = keras.Input(shape=(28, 28, 1))

# Encoder
x = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(input_img)
x = keras.layers.MaxPooling2D(2, padding='same')(x)
x = keras.layers.Conv2D(16, 3, activation='relu', padding='same')(x)
encoded = keras.layers.MaxPooling2D(2, padding='same')(x)

# Decoder
x = keras.layers.Conv2DTranspose(16, 3, activation='relu', padding='same')(encoded)
x = keras.layers.UpSampling2D(2)(x)
x = keras.layers.Conv2DTranspose(32, 3, activation='relu', padding='same')(x)
x = keras.layers.UpSampling2D(2)(x)
decoded = keras.layers.Conv2D(1, 3, activation='sigmoid', padding='same')(x)

autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train autoencoder
autoencoder.fit(
    x_train[..., np.newaxis], x_train[..., np.newaxis],
    epochs=20,
    batch_size=128,
    validation_data=(x_test[..., np.newaxis], x_test[..., np.newaxis])
)

# Create encoder model for feature extraction
encoder = keras.Model(input_img, encoded)

# MLP Classifier [4]
mlp = keras.Sequential([
    keras.layers.Flatten(input_shape=(7,7,16)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(7, activation='sigmoid')
])

mlp.compile(optimizer='adam',
           loss='binary_crossentropy',
           metrics=['accuracy'])

# Train MLP on encoded features
encoded_train = encoder.predict(x_train[..., np.newaxis])
mlp.fit(encoded_train, y_train_seg, epochs=15, batch_size=128)

# Evaluation
encoded_test = encoder.predict(x_test[..., np.newaxis])
predictions = mlp.predict(encoded_test)

# Convert predictions to digits using Hamming distance
def segment_to_digit(pred):
    pred_binary = (pred > 0.5).astype(int)
    distances = []
    for digit in range(10):
        distances.append(np.sum(pred_binary != SEGMENT_MAP[digit]))
    return np.argmin(distances)

y_pred = np.array([segment_to_digit(p) for p in predictions])

# Confusion matrix [6][7]
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
disp.plot(cmap=plt.cm.Blues)
plt.show()

# Performance report
accuracy = np.mean(y_pred == y_test)
print(f"Test Accuracy: {accuracy:.4f}")
print("Class-wise Accuracy:")
for i in range(10):
    class_acc = np.mean(y_pred[y_test == i] == i)
    print(f"Digit {i}: {class_acc:.4f}")
