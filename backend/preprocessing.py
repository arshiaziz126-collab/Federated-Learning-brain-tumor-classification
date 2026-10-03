"""Image preprocessing.

Training and inference share ``_to_array`` so an uploaded MRI is prepared in
exactly the way the training images were. Pixel values stay 0-255 (uint8);
scaling to [0, 1] happens *inside* the model (Rescaling layer), so the saved
model is self-contained.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def _to_array(im: Image.Image, img_size: int, grayscale: bool) -> np.ndarray:
    """PIL image -> uint8 array of shape (img_size, img_size, channels)."""
    if im.mode in ("I", "I;16", "I;16B", "F"):          # 16-bit / float scans
        a = np.asarray(im, dtype=np.float32)
        rng = float(a.max() - a.min())
        a = (a - a.min()) / rng * 255.0 if rng > 0 else np.zeros_like(a)
        im = Image.fromarray(a.astype(np.uint8), mode="L")
    im = im.convert("L" if grayscale else "RGB")
    im = im.resize((img_size, img_size), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.uint8)
    return arr[..., None] if grayscale else arr


def load_image_array(path, img_size: int, grayscale: bool) -> np.ndarray:
    with Image.open(path) as im:
        im.load()
        return _to_array(im, img_size, grayscale)


def load_images(paths, img_size: int, grayscale: bool) -> np.ndarray:
    """Load many images into one uint8 array (N, H, W, C)."""
    channels = 1 if grayscale else 3
    out = np.empty((len(paths), img_size, img_size, channels), dtype=np.uint8)
    for i, p in enumerate(paths):
        out[i] = load_image_array(p, img_size, grayscale)
    return out


def preprocess_pil(im: Image.Image, img_size: int, grayscale: bool) -> np.ndarray:
    """Uploaded image -> float32 batch of one, ready for ``model.predict``."""
    return _to_array(im, img_size, grayscale)[None].astype(np.float32)
