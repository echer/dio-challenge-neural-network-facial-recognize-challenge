import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import mixed_precision
from tensorflow.keras import regularizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import seaborn as sns

mixed_precision.set_global_policy('mixed_float16')

# 🔥 força uso de GPU (se disponível)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU detectada:", gpus)
else:
    print("Rodando em CPU")

IMG_SIZE = 128
BATCH_SIZE = 16
SEED=123
HEAD_EPOCHS = 50
FINE_TUNE_EPOCHS = 70
FINE_TUNE_LAYERS = 25
HEAD_LEARNING_RATE = 3e-4
FINE_TUNE_LEARNING_RATE = 1e-5
FIRST_DROPOUT = 0.35
SECOND_DROPOUT = 0.25
EARLY_STOP_PATIENCE = 12
REDUCE_LR_PATIENCE = 6
CHECKPOINT_MIN_VAL_ACCURACY = 0.70

DATASET_DIR = Path("dataset")
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

if not TRAIN_DIR.exists() or not VAL_DIR.exists():
    raise FileNotFoundError(
        "Estrutura esperada: dataset/train/<classe> e dataset/val/<classe>"
    )

class_names = sorted(path.name for path in TRAIN_DIR.iterdir() if path.is_dir())
val_class_names = sorted(path.name for path in VAL_DIR.iterdir() if path.is_dir())
if class_names != val_class_names:
    raise ValueError(
        f"Classes diferentes entre treino e validacao: "
        f"train={class_names}, val={val_class_names}"
    )

print("Classes:", class_names)


def collect_paths_and_labels(base_dir):
    paths = []
    labels = []
    for label, class_name in enumerate(class_names):
        class_paths = sorted(
            str(path)
            for path in (base_dir / class_name).iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        paths.extend(class_paths)
        labels.extend([label] * len(class_paths))
    return paths, labels


train_paths, train_labels = collect_paths_and_labels(TRAIN_DIR)
val_paths, val_labels = collect_paths_and_labels(VAL_DIR)

for label, class_name in enumerate(class_names):
    train_count = sum(1 for value in train_labels if value == label)
    val_count = sum(1 for value in val_labels if value == label)
    print(
        f"{class_name}: {train_count} treino, "
        f"{val_count} validacao"
    )


def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    return image, label


def make_dataset(paths, labels, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    return (
        ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )


train_ds = make_dataset(train_paths, train_labels, shuffle=True)
val_ds = make_dataset(val_paths, val_labels)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.02),
    layers.RandomTranslation(0.02, 0.02),
    layers.RandomZoom(0.04),
    layers.RandomBrightness(0.02, value_range=(0, 255)),
    layers.RandomContrast(0.04),
])

reg = regularizers.l2(0.0001)

# 🧠 MODELO COM TRANSFER LEARNING
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(FIRST_DROPOUT)(x)
x = layers.Dense(96, activation="relu", kernel_regularizer=reg)(x)
x = layers.Dropout(SECOND_DROPOUT)(x)
outputs = layers.Dense(len(class_names), activation="softmax", dtype="float32")(x)
model = models.Model(inputs, outputs)

# ⚙️ compilação
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=HEAD_LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 🏋️ treino
def make_callbacks():
    reduce_lr = ReduceLROnPlateau(
        monitor='val_accuracy',
        mode='max',
        factor=0.5,
        patience=REDUCE_LR_PATIENCE,
        min_lr=1e-6
    )
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        mode='max',
        patience=EARLY_STOP_PATIENCE,
        min_delta=0.001,
        restore_best_weights=True
    )
    checkpoint = ModelCheckpoint(
        "modelo_faces.keras",
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        initial_value_threshold=CHECKPOINT_MIN_VAL_ACCURACY,
    )
    return [early_stop, reduce_lr, checkpoint]

print(
    "Training config:",
    {
        "head_lr": HEAD_LEARNING_RATE,
        "fine_tune_lr": FINE_TUNE_LEARNING_RATE,
        "fine_tune_layers": FINE_TUNE_LAYERS,
        "dropout": [FIRST_DROPOUT, SECOND_DROPOUT],
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "reduce_lr_patience": REDUCE_LR_PATIENCE,
    },
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=HEAD_EPOCHS,
    callbacks=make_callbacks(),
)

# Ajuste fino das últimas camadas do MobileNetV2 com learning rate menor.
base_model.trainable = True
for layer in base_model.layers[:-FINE_TUNE_LAYERS]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

fine_tune_history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=HEAD_EPOCHS + FINE_TUNE_EPOCHS,
    initial_epoch=len(history.history["loss"]),
    callbacks=make_callbacks(),
)

# 💾 carregar o melhor checkpoint salvo
model = tf.keras.models.load_model("modelo_faces.keras")

# salvar classes
with open("classes.txt", "w") as f:
    for nome in class_names:
        f.write(nome + "\n")

# 🔮 previsões
preds = model.predict(val_ds)
y_pred = np.argmax(preds, axis=1)
y_true = np.concatenate([y for x, y in val_ds], axis=0)
pred_counts = np.bincount(y_pred, minlength=len(class_names))
print("Predicoes por classe:", dict(zip(class_names, map(int, pred_counts))))

# 📊 matriz de confusão
cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predito")
plt.ylabel("Real")
plt.title("Matriz de Confusão")
plt.savefig("confusion_matrix.png")

# 📄 relatório completo
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
