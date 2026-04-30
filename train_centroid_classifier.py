from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model, load_model


IMG_SIZE = 128
DATASET_DIR = Path("dataset")
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_paths(class_dir):
    return sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Imagem invalida: {path}")
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    return image


def build_embedder():
    trained_model = load_model("modelo_faces.keras")
    return Model(
        inputs=trained_model.input,
        outputs=trained_model.get_layer("global_average_pooling2d").output,
    )


def embed_images(embedder, paths):
    images = np.stack([load_image(path) for path in paths], axis=0)
    embeddings = embedder.predict(images, verbose=0)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)


def predict_from_centroids(embeddings, centroids):
    scores = embeddings @ centroids.T
    pred_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)
    return pred_ids, confidences, scores


def main():
    class_names = sorted(path.name for path in TRAIN_DIR.iterdir() if path.is_dir())
    print("Classes:", class_names)

    embedder = build_embedder()

    centroids = []
    for class_name in class_names:
        paths = image_paths(TRAIN_DIR / class_name)
        embeddings = embed_images(embedder, paths)
        centroid = embeddings.mean(axis=0)
        centroid = centroid / np.maximum(np.linalg.norm(centroid), 1e-12)
        centroids.append(centroid)
        print(f"{class_name}: {len(paths)} treino")

    centroids = np.stack(centroids, axis=0)
    np.savez("face_centroids.npz", class_names=np.array(class_names), centroids=centroids)

    val_paths = []
    y_true = []
    for label, class_name in enumerate(class_names):
        paths = image_paths(VAL_DIR / class_name)
        val_paths.extend(paths)
        y_true.extend([label] * len(paths))
        print(f"{class_name}: {len(paths)} validacao")

    val_embeddings = embed_images(embedder, val_paths)
    y_pred, confidences, scores = predict_from_centroids(val_embeddings, centroids)
    y_true = np.array(y_true)

    print("Predicoes por classe:", {
        class_name: int(np.sum(y_pred == index))
        for index, class_name in enumerate(class_names)
    })
    print("Matriz de confusao:")
    print(confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names))))
    print("\n=== Classification Report ===\n")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=np.arange(len(class_names)),
            target_names=class_names,
            zero_division=0,
        )
    )

    for path, true_id, pred_id, confidence, score_row in zip(
        val_paths, y_true, y_pred, confidences, scores
    ):
        if true_id == pred_id:
            continue
        probs = dict(zip(class_names, map(float, score_row)))
        print(
            f"{path}: real={class_names[true_id]}, "
            f"pred={class_names[pred_id]} ({confidence:.2f}) | {probs}"
        )


if __name__ == "__main__":
    main()
