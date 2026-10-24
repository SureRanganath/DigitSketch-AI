import matplotlib
 
matplotlib.use("Agg")  # save plots to files without needing a display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
 
MODEL_PATH = "mnist_cnn.keras"
IMG_SIZE = 28
EPOCHS = 8
BATCH_SIZE = 128
 
 
def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        # light augmentation (training only) for hand-drawn robustness
        tf.keras.layers.RandomTranslation(0.1, 0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
 
 
def main():
    # 1. Load and preprocess the data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = (x_train / 255.0).astype("float32")[..., None]
    x_test = (x_test / 255.0).astype("float32")[..., None]
    print("Train:", x_train.shape, "| Test:", x_test.shape)
 
    # 2. Build, compile, train
    model = build_model()
    model.summary()
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )
 
    # 3. Evaluate on the held-out test set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest loss:     {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
 
    # 4. Save the model for the Gradio app
    model.save(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")
 
    # 5. Training curves
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
    plt.close(fig)
 
    # 6. Sample predictions
    preds = np.argmax(model.predict(x_test[:10], verbose=0), axis=1)
    print("Predicted:", preds)
    print("Actual:   ", y_test[:10])
 
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(x_test[i].reshape(IMG_SIZE, IMG_SIZE), cmap="gray")
        ax.set_title(f"Pred: {preds[i]} / True: {y_test[i]}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("sample_predictions.png")
    plt.close(fig)
 
 
if __name__ == "__main__":
    main()
 