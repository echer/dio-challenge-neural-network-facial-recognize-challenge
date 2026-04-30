from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.models import load_model


IMG_SIZE = 128
CONFIDENCE_THRESHOLD = 0.9
MIN_CONFIDENCE_GAP = 0.4
FACE_MARGIN = 0.05
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

TEST_DIR = Path("dataset/test")
OUTPUT_DIR = Path("dataset/test_output")


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

    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype(np.float32)
    return np.expand_dims(face, axis=0)


def classify_face(model, class_names, face):
    pred = model.predict(face, verbose=0)[0]
    class_id = int(np.argmax(pred))
    confidence = float(pred[class_id])
    sorted_confidences = np.sort(pred)
    confidence_gap = float(sorted_confidences[-1] - sorted_confidences[-2])

    if confidence < CONFIDENCE_THRESHOLD or confidence_gap < MIN_CONFIDENCE_GAP:
        return "desconhecido", confidence, pred

    return class_names[class_id], confidence, pred


def draw_label(image, label, x, y, w, h):
    color = (0, 255, 0) if not label.startswith("desconhecido") else (0, 200, 255)
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    text_y = max(25, y - 10)
    cv2.putText(
        image,
        label,
        (x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )


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

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"{image_path}: arquivo invalido")
            continue

        faces = detect_faces(image, face_cascade)
        print(f"\n{image_path.name}: {len(faces)} rosto(s)")

        for index, (x, y, w, h) in enumerate(faces, start=1):
            face = crop_face(image, x, y, w, h)
            if face is None:
                continue

            name, confidence, pred = classify_face(model, class_names, face)
            label = f"{name} ({confidence:.2f})"
            draw_label(image, label, x, y, w, h)

            probs = dict(zip(class_names, map(float, pred)))
            print(f"  face {index}: {label} | {probs}")

        output_path = OUTPUT_DIR / image_path.name
        cv2.imwrite(str(output_path), image)
        print(f"  salvo: {output_path}")


if __name__ == "__main__":
    main()
