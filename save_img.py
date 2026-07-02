import os
from datetime import datetime

BASE_DIR = "captured_images"

REAL_DIR = os.path.join(BASE_DIR, "real")
FAKE_DIR = os.path.join(BASE_DIR, "fake")

os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(FAKE_DIR, exist_ok=True)


def save(img, is_fake):

    folder = FAKE_DIR if is_fake else REAL_DIR

    filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"

    filepath = os.path.join(folder, filename)

    img.save(filepath)

    return filepath