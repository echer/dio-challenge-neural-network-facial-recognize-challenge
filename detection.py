import cv2
import numpy as np
from tensorflow.keras.models import load_model

# carregar modelo
model = load_model("modelo_faces.keras")

# carregar nomes
with open("classes.txt") as f:
    class_names = [linha.strip() for linha in f]

# detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)

IMG_SIZE = 128
CONFIDENCE_THRESHOLD = 0.85
MIN_CONFIDENCE_GAP = 0.35
FACE_MARGIN = 0.05
DEBUG_PROBS = False


def crop_face(frame, x, y, w, h):
    center_x = x + (w / 2)
    center_y = y + (h / 2)
    crop_size = int(max(w, h) * (1 + (2 * FACE_MARGIN)))
    x1 = max(0, int(center_x - (crop_size / 2)))
    y1 = max(0, int(center_y - (crop_size / 2)))
    x2 = min(frame.shape[1], x1 + crop_size)
    y2 = min(frame.shape[0], y1 + crop_size)
    x1 = max(0, x2 - crop_size)
    y1 = max(0, y2 - crop_size)

    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return None

    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype(np.float32)
    return face


def classify_prediction(pred):
    class_id = int(np.argmax(pred))
    confidence = float(pred[class_id])
    sorted_confidences = np.sort(pred)
    confidence_gap = float(sorted_confidences[-1] - sorted_confidences[-2])

    if DEBUG_PROBS:
        print(dict(zip(class_names, map(float, pred))))

    if confidence < CONFIDENCE_THRESHOLD or confidence_gap < MIN_CONFIDENCE_GAP:
        return "desconhecido", confidence

    return class_names[class_id], confidence

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # 🔴 DETECÇÃO
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=7,
        minSize=(80, 80),
    )

    face_boxes = []
    face_images = []
    for (x, y, w, h) in faces:
        if w < 50 or h < 50:
            continue

        face = crop_face(frame, x, y, w, h)
        if face is None:
            continue

        face_boxes.append((x, y, w, h))
        face_images.append(face)

    if face_images:
        preds = model.predict(np.stack(face_images, axis=0), verbose=0)
        for (x, y, w, h), pred in zip(face_boxes, preds):
            nome, confianca = classify_prediction(pred)
            label = f"{nome} ({confianca:.2f})"

            # 🟣 desenhar
            color = (0, 255, 0) if nome != "desconhecido" else (0, 200, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, max(25, y-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Reconhecimento Facial", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
