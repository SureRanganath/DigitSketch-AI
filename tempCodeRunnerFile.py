import os
import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = "mnist_cnn.keras"
IMG_SIZE = 28  # MNIST image size (28 x 28, grayscale)
DIGIT_BOX = 20  # MNIST digits are scaled to fit a 20x20 box inside the 28x28 frame

# ---------------------------------------------------------------
# 1. Model: load if saved, otherwise train a small CNN
# ---------------------------------------------------------------
def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        # light augmentation (only active during training) for hand-drawn robustness
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


def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        print(f"Loading saved model from {MODEL_PATH}")
        return tf.keras.models.load_model(MODEL_PATH)

    print("No saved model found. Training a new one...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = (x_train / 255.0).astype("float32")[..., None]
    x_test = (x_test / 255.0).astype("float32")[..., None]

    model = build_model()
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(x_train, y_train, validation_split=0.1, epochs=8, batch_size=128)
    _, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {acc:.4f}")
    model.save(MODEL_PATH)
    return model


model = load_or_train_model()


# ---------------------------------------------------------------
# 2. Preprocessing: turn a drawing into an MNIST-style 28x28 image
# ---------------------------------------------------------------
def extract_strokes(sketch):
    """Return a 2D float array (0-255) with WHITE strokes on a BLACK background."""
    if sketch is None:
        return None

    comp = None
    if isinstance(sketch, dict):
        # Gradio 4/5 Sketchpad: use the stroke layers' alpha channel when available
        layers = [
            l for l in (sketch.get("layers") or [])
            if isinstance(l, np.ndarray) and l.ndim == 3 and l.shape[2] == 4
        ]
        if layers:
            alpha = np.max([l[..., 3] for l in layers], axis=0).astype(np.float32)
            if alpha.max() > 0:
                return alpha
        comp = sketch.get("composite")
    else:
        comp = sketch

    if comp is None:
        return None
    comp = np.asarray(comp)

    # Transparent background with strokes: alpha channel is the drawing
    if comp.ndim == 3 and comp.shape[2] == 4:
        alpha = comp[..., 3].astype(np.float32)
        if alpha.min() == 0 and alpha.max() > 0:
            return alpha
        comp = comp[..., :3]

    # Otherwise convert to grayscale and make strokes white on black
    gray = comp.mean(axis=2) if comp.ndim == 3 else comp.astype(np.float32)
    gray = gray.astype(np.float32)
    if gray.mean() > 127:  # light background, dark ink -> invert
        gray = 255.0 - gray
    return gray


def preprocess(sketch):
    """
    Match the MNIST format:
      - grayscale, white digit on black background
      - crop to the digit, scale to fit a 20x20 box (keep aspect ratio)
      - paste into a 28x28 frame and center by center of mass
      - pixel values scaled to 0-1
    Returns a (28, 28) float32 array, or None if nothing was drawn.
    """
    img = extract_strokes(sketch)
    if img is None or img.max() < 10:
        return None

    ys, xs = np.where(img > 20)
    if len(ys) == 0:
        return None
    crop = img[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]

    h, w = crop.shape
    scale = DIGIT_BOX / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = Image.fromarray(np.clip(crop, 0, 255).astype(np.uint8)).resize(
        (new_w, new_h), Image.LANCZOS
    )

    frame = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    top, left = (IMG_SIZE - new_h) // 2, (IMG_SIZE - new_w) // 2
    frame[top: top + new_h, left: left + new_w] = np.array(resized, dtype=np.float32)

    # Shift so the digit's center of mass sits at the center of the frame
    total = frame.sum()
    if total > 0:
        cy = (frame.sum(axis=1) * np.arange(IMG_SIZE)).sum() / total
        cx = (frame.sum(axis=0) * np.arange(IMG_SIZE)).sum() / total
        shift_y, shift_x = int(round(IMG_SIZE / 2 - cy)), int(round(IMG_SIZE / 2 - cx))
        frame = np.roll(frame, (shift_y, shift_x), axis=(0, 1))

    return (frame / 255.0).astype(np.float32)

# ---------------------------------------------------------------
# 3. Prediction
# ---------------------------------------------------------------
def predict(sketch):
    processed = preprocess(sketch)
    if processed is None:
        return None, None

    probs = model.predict(processed.reshape(1, IMG_SIZE, IMG_SIZE, 1), verbose=0)[0]
    labels = {str(i): float(p) for i, p in enumerate(probs)}

    # Enlarged preview of exactly what the model sees
    preview = Image.fromarray((processed * 255).astype(np.uint8)).resize(
        (140, 140), Image.NEAREST
    )
    return labels, preview


# ---------------------------------------------------------------
# 4. Gradio interface
# ---------------------------------------------------------------
with gr.Blocks(title="DigitSketch-Ai   -Handwritten Digit Recognizer") as demo:
    gr.Markdown(
        "# Handwritten Digit Recognizer\n"
        "Draw a single digit (0-9) on the canvas. The prediction updates as you draw."
    )
    with gr.Row():
        with gr.Column():
            sketch = gr.Sketchpad(
                label="Draw a digit",
                type="numpy",
                canvas_size=(280, 280),
                brush=gr.Brush(default_size=18, colors=["#000000"], color_mode="fixed"),
            )
            clear_btn = gr.ClearButton(components=[sketch], value="Clear")
        with gr.Column():
            label_out = gr.Label(num_top_classes=3, label="Prediction")
            preview_out = gr.Image(
                label="What the model sees (28x28 grayscale)",
                width=160,
                height=160,
                interactive=False,
            )

    sketch.change(predict, inputs=sketch, outputs=[label_out, preview_out])

if __name__ == "__main__":
    demo.launch()  # use demo.launch(share=True) for a temporary public link
