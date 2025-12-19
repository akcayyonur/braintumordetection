import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report

# -----------------------
# Paths
# -----------------------
DATA_DIR = "data/brain_tumor_multiclass"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "brain_tumor_multiclass.h5")
META_PATH = os.path.join(MODEL_DIR, "meta.json")

# -----------------------
# Hyperparams
# -----------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# Stage 1 (head training)
EPOCHS_STAGE1 = 12
LR_STAGE1 = 1e-4

# Stage 2 (fine-tuning)
EPOCHS_STAGE2 = 15
LR_STAGE2 = 5e-6

# Fine-tune more layers to handle domain shift (dataset1 vs dataset2)
UNFREEZE_LAST_N_LAYERS = 60  # <- kritik: 60 layer açık

# -----------------------
# Data Generator (domain-robust augmentation)
# -----------------------
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.25,
    width_shift_range=0.08,
    height_shift_range=0.08,
    shear_range=0.10,
    brightness_range=(0.7, 1.3),
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="training",
    class_mode="categorical",
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="validation",
    class_mode="categorical",
    shuffle=False
)

NUM_CLASSES = train_gen.num_classes

# -----------------------
# Class weights (still good practice)
# -----------------------
y_classes = train_gen.classes
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_classes),
    y=y_classes
)
class_weights = dict(enumerate(class_weights))

# -----------------------
# Build model
# -----------------------
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.45)(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=outputs)

# -----------------------
# Stage 1: train head
# -----------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR_STAGE1),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_stage1 = [
    EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.3, min_lr=1e-6),
    ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True)
]

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_STAGE1,
    class_weight=class_weights,
    callbacks=callbacks_stage1
)

# -----------------------
# Stage 2: fine-tune last N layers of backbone
# -----------------------
base_model.trainable = True

# Freeze all except last N layers
if UNFREEZE_LAST_N_LAYERS > 0:
    for layer in base_model.layers[:-UNFREEZE_LAST_N_LAYERS]:
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR_STAGE2),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_stage2 = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.3, min_lr=1e-7),
    ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True)
]

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_STAGE2,
    class_weight=class_weights,
    callbacks=callbacks_stage2
)

# -----------------------
# Save meta
# -----------------------
with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(
        {
            "class_indices": train_gen.class_indices,
            "img_size": IMG_SIZE[0],
            "notes": {
                "augmentation": "domain-robust (brightness/zoom/shear/shift/rot)",
                "fine_tune_last_n_layers": UNFREEZE_LAST_N_LAYERS,
                "lr_stage1": LR_STAGE1,
                "lr_stage2": LR_STAGE2
            }
        },
        f,
        indent=2
    )

print("\n✅ Eğitim tamamlandı.")
print("✅ Model:", MODEL_PATH)
print("✅ Meta :", META_PATH)
print("Sınıflar:", train_gen.class_indices)

# -----------------------
# Quick eval: confusion matrix + report on validation split
# -----------------------
val_gen.reset()
preds = model.predict(val_gen, verbose=0)
y_pred = np.argmax(preds, axis=1)
y_true = val_gen.classes

print("\n--- CONFUSION MATRIX (VAL) ---")
print(confusion_matrix(y_true, y_pred))

print("\n--- CLASSIFICATION REPORT (VAL) ---")
print(classification_report(y_true, y_pred, target_names=list(train_gen.class_indices.keys())))
