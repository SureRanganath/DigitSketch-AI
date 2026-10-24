import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# ---------------------------------------------------------------
# 1. Load the data (60,000 train / 10,000 test images, 28x28)
# ---------------------------------------------------------------
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
print("Train shape:", X_train.shape, "| Test shape:", X_test.shape)

# ---------------------------------------------------------------
# 2. Preprocess
# ---------------------------------------------------------------
# Scale pixel values from 0-255 to 0-1
X_train = X_train / 255.0
X_test = X_test / 255.0

# Flatten each 28x28 image into a 784-length vector
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# One-hot encode the labels
num_classes = 10
y_train_oh = to_categorical(y_train, num_classes)
y_test_oh = to_categorical(y_test, num_classes)

# ---------------------------------------------------------------
# 3. Build the model
# ---------------------------------------------------------------
input_dims = X_train.shape[1]

model = Sequential([
    Dense(512, activation="sigmoid", input_shape=(input_dims,), name="Hidden_layer_1"),
    Dense(128, activation="sigmoid", name="Hidden_layer_2"),
    Dense(num_classes, activation="softmax", name="output_layer"),
])
model.summary()

# ---------------------------------------------------------------
# 4. Compile and train
# ---------------------------------------------------------------
model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)

history = model.fit(
    X_train, y_train_oh,
    validation_split=0.2,
    epochs=20,
    batch_size=128,
    verbose=1,
)

# ---------------------------------------------------------------
# 5. Evaluate on the test set
# ---------------------------------------------------------------
test_loss, test_acc = model.evaluate(X_test, y_test_oh, verbose=0)
print(f"\nTest loss:     {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")

# ---------------------------------------------------------------
# 6. Plot training curves
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history["accuracy"], label="train")
axes[0].plot(history.history["val_accuracy"], label="validation")
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(history.history["loss"], label="train")
axes[1].plot(history.history["val_loss"], label="validation")
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].legend()

plt.tight_layout()
plt.savefig("training_curves.png")
plt.show()

# ---------------------------------------------------------------
# 7. Predict a few test images
# ---------------------------------------------------------------
preds = np.argmax(model.predict(X_test[:10], verbose=0), axis=1)
print("Predicted:", preds)
print("Actual:   ", y_test[:10])

fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(28, 28), cmap="gray")
    ax.set_title(f"Pred: {preds[i]} / True: {y_test[i]}")
    ax.axis("off")
plt.tight_layout()
plt.savefig("sample_predictions.png")
plt.show()

# Optionally save the trained model
# model.save("mnist_mlp.keras")