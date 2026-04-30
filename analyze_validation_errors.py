from pathlib import Path
import shutil

import cv2
import numpy as np
from tensorflow.keras.models import load_model


IMG_SIZE = 128
VAL_DIR = Path("dataset/val")
OUTPUT_DIR = Path("dataset/val_errors")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_face(path):
    image = cv2.imread(str(path))
    if image is None:
        return None

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    return np.expand_dims(image, axis=0)


def main():
    model = load_model("modelo_faces.keras")
    with open("classes.txt") as f:
        class_names = [line.strip() for line in f if line.strip()]

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    wrong = 0
    for true_name in class_names:
        for path in sorted((VAL_DIR / true_name).iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            face = load_face(path)
            if face is None:
                continue

            pred = model.predict(face, verbose=0)[0]
            pred_id = int(np.argmax(pred))
            pred_name = class_names[pred_id]
            confidence = float(pred[pred_id])
            total += 1

            if pred_name == true_name:
                continue

            wrong += 1
            probs = dict(zip(class_names, map(float, pred)))
            print(
                f"{path}: real={true_name}, pred={pred_name} "
                f"({confidence:.2f}) | {probs}"
            )

            output_name = (
                f"real-{true_name}_pred-{pred_name}_{confidence:.2f}_"
                f"{path.stem}.jpg"
            )
            image = cv2.imread(str(path))
            cv2.imwrite(str(OUTPUT_DIR / output_name), image)

    print(f"\nErros: {wrong}/{total}")
    print(f"Crops errados copiados para: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
