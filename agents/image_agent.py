"""
image_agent.py
وكيل الصور: يستقبل صورة أشعة صدر ويُرجع احتمال الالتهاب الرئوي (0 إلى 1).
NORMAL = 0 ، PNEUMONIA = 1  ->  المخرج = P(PNEUMONIA)
"""
import os
import numpy as np
from PIL import Image

MODEL_PATH = os.environ.get("CNN_MODEL_PATH", "cnn_pneumonia.keras")
IMG_SIZE = (224, 224)
THRESHOLD = 0.50   # 0.23 = العتبة المضبوطة على val لرفع Recall
CLASS_NAMES = {0: "NORMAL", 1: "PNEUMONIA"}

_model = None


def _get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"ملف النموذج غير موجود: {MODEL_PATH}")
        import tensorflow as tf
        _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return _model


def _load_image(image):
    if isinstance(image, (str, os.PathLike)):
        img = Image.open(image)
    elif isinstance(image, Image.Image):
        img = image
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image.astype("uint8"))
    else:
        img = Image.open(image)
    return img.convert("RGB")


def _preprocess(image):
    img = _load_image(image).resize(IMG_SIZE, Image.NEAREST)
    arr = np.asarray(img, dtype="float32")
    return arr[None, ...]


def predict(image) -> float:
    """يُرجع احتمال الالتهاب الرئوي بين 0 و1 (قيمة خام بدون عتبة)."""
    prob = _get_model().predict(_preprocess(image), verbose=0)
    return float(prob.ravel()[0])


def predict_label(image, threshold=None) -> str:
    thr = THRESHOLD if threshold is None else threshold
    return CLASS_NAMES[int(predict(image) >= thr)]


def predict_full(image, threshold=None) -> dict:
    thr = THRESHOLD if threshold is None else threshold
    p = predict(image)
    return {"probability": p, "label": CLASS_NAMES[int(p >= thr)], "threshold": thr}
