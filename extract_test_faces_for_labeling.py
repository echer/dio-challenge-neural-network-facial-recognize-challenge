from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.models import load_model


IMG_SIZE = 128
FACE_MARGIN = 0.05
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

TEST_DIR = Path("dataset/test")
OUTPUT_DIR = Path("dataset/review_faces")


def detect_faces(image, face_cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(100, 100),
    )


def crop_face(image, x, y, w, h):
    center_x = x + (w / 2)
    center_y = y + (h / 2)
    crop_size = int(max(w, h) * (1 + (2 * FACE_MARGIN)))
    x1 = max(0, int(center_x - (crop_size / 2)))
    y1 = max(0, int(center_y - (crop_size / 2)))
    x2 = min(image.shape[1], x1 + crop_size)
    y2 = min(image.shape[0], y1 + crop_size)
    x1 = max(0, x2 - crop_size)
    y1 = max(0, y2 - crop_size)

    face = image[y1:y2, x1:x2]
    if face.size == 0:
        return None
    return cv2.resize(face, (IMG_SIZE, IMG_SIZE))


def main():
    model = load_model("modelo_faces.keras")
    with open("classes.txt") as f:
        class_names = [line.strip() for line in f if line.strip()]

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path
        for path in TEST_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    saved = 0
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        faces = detect_faces(image, face_cascade)
        for index, (x, y, w, h) in enumerate(faces, start=1):
            face_bgr = crop_face(image, x, y, w, h)
            if face_bgr is None:
                continue

            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
            pred = model.predict(np.expand_dims(face_rgb, axis=0), verbose=0)[0]
            class_id = int(np.argmax(pred))
            confidence = float(pred[class_id])
            predicted_name = class_names[class_id]

            output_name = (
                f"{image_path.stem}_face{index:02d}_"
                f"pred-{predicted_name}_{confidence:.2f}.jpg"
            )
            cv2.imwrite(str(OUTPUT_DIR / output_name), face_bgr)
            saved += 1

    print(f"Faces extraidas para revisao: {saved}")
    print(f"Pasta: {OUTPUT_DIR}")
    print("Mova cada imagem revisada para dataset/train/<classe> ou dataset/val/<classe>.")


if __name__ == "__main__":
    main()
