import argparse
from pathlib import Path

import cv2


IMG_SIZE = 128
FACE_MARGIN = 0.05
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DETECTOR_SETTINGS = [
    {"scaleFactor": 1.1, "minNeighbors": 7, "minSize": (80, 80)},
    {"scaleFactor": 1.05, "minNeighbors": 5, "minSize": (50, 50)},
    {"scaleFactor": 1.03, "minNeighbors": 3, "minSize": (30, 30)},
]


def rotate_image(image, rotation):
    if rotation == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def detect_faces(gray, face_cascade):
    for settings in DETECTOR_SETTINGS:
        faces = face_cascade.detectMultiScale(gray, **settings)
        if len(faces) > 0:
            return faces
    return []


def crop_largest_face(image, face_cascade):
    best_face = None
    best_area = 0

    for rotation in (0, 90, 180, 270):
        rotated = rotate_image(image, rotation)
        cropped = crop_largest_face_without_rotation(rotated, face_cascade)
        if cropped is None:
            continue

        area = cropped.shape[0] * cropped.shape[1]
        if area > best_area:
            best_face = cropped
            best_area = area

    return best_face


def crop_largest_face_without_rotation(image, face_cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = detect_faces(gray, face_cascade)

    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda face_box: face_box[2] * face_box[3])
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


def image_paths(directory):
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def crop_class_folder(input_dir, output_dir, class_name, face_cascade):
    output_class_dir = output_dir / class_name
    output_class_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped = 0
    for path in image_paths(input_dir):
        image = cv2.imread(str(path))
        if image is None:
            skipped += 1
            print(f"Pulando arquivo invalido: {path}")
            continue

        face = crop_largest_face(image, face_cascade)
        if face is None:
            skipped += 1
            print(f"Nenhum rosto detectado: {path}")
            continue

        output_path = output_class_dir / f"{path.stem}.jpg"
        cv2.imwrite(str(output_path), face)
        saved += 1

    print(f"{class_name}: {saved} salvas, {skipped} ignoradas")


def main():
    parser = argparse.ArgumentParser(
        description="Recorta rostos de fotos reais no mesmo formato do dataset."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Pasta com fotos ou com subpastas por classe.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Pasta de saida para os rostos recortados.",
    )
    parser.add_argument(
        "--class-name",
        help="Use quando input_dir contem fotos de uma unica pessoa.",
    )
    args = parser.parse_args()

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    if args.class_name:
        crop_class_folder(args.input_dir, args.output_dir, args.class_name, face_cascade)
        return

    class_dirs = sorted(path for path in args.input_dir.iterdir() if path.is_dir())
    if not class_dirs:
        raise SystemExit(
            "Nenhuma subpasta de classe encontrada. "
            "Use --class-name se input_dir tiver fotos de uma unica pessoa."
        )

    for class_dir in class_dirs:
        crop_class_folder(class_dir, args.output_dir, class_dir.name, face_cascade)


if __name__ == "__main__":
    main()
