import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve, roc_auc_score

DATA_DIR = "data/brain_tumor_dataset"
MODEL_PATH = "models/brain_tumor_mobilenetv2.h5"
META_PATH = "models/meta.json"
IMG_SIZE = 224
BATCH_SIZE = 16


def main():
    model = tf.keras.models.load_model(MODEL_PATH)

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    val_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation",
        shuffle=False
    )

    y_true = val_gen.classes
    y_pred = model.predict(val_gen).ravel()

    auc = roc_auc_score(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    # Youden's J statistic
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[best_idx]

    # Gri alan için ikinci eşikler
    low_threshold = max(optimal_threshold - 0.15, 0.1)
    high_threshold = min(optimal_threshold + 0.15, 0.9)

    result = {
        "auc": float(auc),
        "low_threshold": float(low_threshold),
        "high_threshold": float(high_threshold)
    }

    with open("models/thresholds.json", "w") as f:
        json.dump(result, f, indent=2)

    print("\n✅ MODEL DEĞERLENDİRME TAMAMLANDI")
    print(f"AUC: {auc:.3f}")
    print(f"Low Threshold : {low_threshold:.2f}")
    print(f"High Threshold: {high_threshold:.2f}")
    print("Kaydedildi → models/thresholds.json")


if __name__ == "__main__":
    main()
