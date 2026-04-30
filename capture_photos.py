import cv2
import os
import time

nome = input("Digite o nome da pessoa: ")
pasta = f"dataset/train/{nome}"

os.makedirs(pasta, exist_ok=True)

max_fotos = 200
face_margin = 0.05

# 🔢 obter índices existentes
arquivos = [f for f in os.listdir(pasta) if f.endswith(".jpg")]

numeros_existentes = sorted([
    int(f.split(".")[0]) for f in arquivos if f.split(".")[0].isdigit()
])

# 🔍 encontrar índices faltantes
faltantes = [i for i in range(max_fotos) if i not in numeros_existentes]

# 🔢 próximo índice sequencial (se não houver faltantes)
proximo = max(numeros_existentes)+1 if numeros_existentes else 0

print(f"Imagens existentes: {len(numeros_existentes)}")
print(f"Faltando: {len(faltantes)}")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)

ultimo_save = 0
intervalo = 0.3

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:

        tempo_atual = time.time()
        if tempo_atual - ultimo_save < intervalo:
            continue

        center_x = x + (w / 2)
        center_y = y + (h / 2)
        crop_size = int(max(w, h) * (1 + (2 * face_margin)))
        x1 = max(0, int(center_x - (crop_size / 2)))
        y1 = max(0, int(center_y - (crop_size / 2)))
        x2 = min(frame.shape[1], x1 + crop_size)
        y2 = min(frame.shape[0], y1 + crop_size)
        x1 = max(0, x2 - crop_size)
        y1 = max(0, y2 - crop_size)

        rosto = frame[y1:y2, x1:x2]

        if rosto.size == 0:
            continue

        rosto = cv2.resize(rosto, (128,128))

        # 🎯 escolher índice correto
        if faltantes:
            indice = faltantes.pop(0)  # preenche buracos primeiro
        else:
            indice = proximo
            proximo += 1

        caminho = f"{pasta}/{indice}.jpg"
        cv2.imwrite(caminho, rosto)

        ultimo_save = tempo_atual

        total_atual = len(os.listdir(pasta))
        print(f"Salvo: {indice}.jpg | Total: {total_atual}/{max_fotos}")

        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

        if total_atual >= max_fotos:
            break

    cv2.imshow("Coletando Faces", frame)

    if len(os.listdir(pasta)) >= max_fotos or cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

print("Dataset completo!")
