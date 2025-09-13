import os
import random
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models, callbacks, regularizers, initializers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt

# ==== CONFIG ====
DATASET_DIR   = "dataset"
IMG_SIZE      = 128     # keep fixed-size to avoid geometric distortions
BATCH_SIZE    = 32
EPOCHS        = 30
VAL_SPLIT     = 0.15    # split inside "train" for validation
SEED          = 42
USE_FLIPS     = False   # set True to allow horizontal/vertical flips (safe-ish)
LEARNING_RATE = 1e-4
L2            = 1e-4    # small weight decay to reduce overfitting

# Determinism (best effort)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ==== DATA GENERATORS ====
# In steganalysis we avoid strong geometric/color augments; flips are the safest option.
if USE_FLIPS:
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=VAL_SPLIT
    )
else:
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=VAL_SPLIT
    )

val_datagen = ImageDataGenerator(rescale=1./255, validation_split=VAL_SPLIT)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory=os.path.join(DATASET_DIR, "train"),
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode='binary',
    seed=SEED,
    shuffle=True,
    subset='training'
)

val_generator = val_datagen.flow_from_directory(
    directory=os.path.join(DATASET_DIR, "train"),
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode='binary',
    seed=SEED,
    shuffle=False,
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    directory=os.path.join(DATASET_DIR, "test"),
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode='binary',
    seed=SEED,
    shuffle=False
)

# ==== CLASS WEIGHTS (helpful if the classes are slightly imbalanced) ====
# ImageDataGenerator exposes .classes (0/1) for all samples in the subset.
def compute_class_weights(gen):
    # Count per class in generator’s underlying index
    labels = gen.classes
    n_total = len(labels)
    n_pos = np.sum(labels)
    n_neg = n_total - n_pos
    # Inverse-frequency weighting
    w0 = n_total / (2.0 * n_neg) if n_neg > 0 else 1.0
    w1 = n_total / (2.0 * n_pos) if n_pos > 0 else 1.0
    return {0: w0, 1: w1}

class_weights = compute_class_weights(train_generator)

# ==== HIGH-PASS FILTER (SRM-like) + TLU LAYER ====
# A common 5x5 SRM residual kernel (KV filter). Scaled for numeric stability.
# Ref patterns widely used in spatial steganalysis front-ends.
HPF_KERNEL = np.array([
    [0,  0, -1,  0,  0],
    [0, -1,  2, -1,  0],
    [-1, 2, -4,  2, -1],
    [0, -1,  2, -1,  0],
    [0,  0, -1,  0,  0]
], dtype=np.float32)

# Normalize kernel a bit to keep activations in sensible range after rescale(0..1)
HPF_KERNEL = HPF_KERNEL / 4.0
HPF_KERNEL = HPF_KERNEL.reshape((5, 5, 1, 1))  # (kh, kw, in_ch, out_ch)

def build_model(img_size=IMG_SIZE):
    inp = layers.Input(shape=(img_size, img_size, 1), name="input")

    # Fixed High-Pass filter front-end
    hpf = layers.Conv2D(
        filters=1, kernel_size=(5, 5), padding='same', use_bias=False,
        kernel_initializer=initializers.Constant(HPF_KERNEL),
        trainable=False, name="fixed_hpf"
    )(inp)

    # TLU clamp (truncate to [-3, 3]) — helps emphasize weak residuals
    tlu = layers.Lambda(lambda x: tf.clip_by_value(x, -3.0, 3.0), name="tlu")(hpf)

    # Feature extractor (compact CNN; L2 regularization)
    x = layers.Conv2D(32, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(L2), activation='relu')(tlu)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(L2), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(L2), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(256, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(L2), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(L2))(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation='sigmoid', name="stego_prob")(x)

    model = models.Model(inputs=inp, outputs=out, name="HPF_TLU_StegoCNN")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    return model

model = build_model()
model.summary()

# ==== CALLBACKS ====
es  = callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=6, restore_best_weights=True)
rlr = callbacks.ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=3, min_lr=1e-6)
ckp = callbacks.ModelCheckpoint("best_model.keras", monitor='val_auc', mode='max',
                                save_best_only=True, save_weights_only=False)

# ==== TRAIN ====
steps_per_epoch = len(train_generator)
validation_steps = len(val_generator)

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_generator,
    validation_steps=validation_steps,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[es, rlr, ckp],
    verbose=1
)

# ==== EVALUATION (TEST) ====
# Predict on test set
test_steps = len(test_generator)
probs = model.predict(test_generator, steps=test_steps, verbose=1).ravel()
y_true = test_generator.classes.astype(int)
y_pred = (probs >= 0.5).astype(int)

# Metrics
acc = accuracy_score(y_true, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
try:
    auc = roc_auc_score(y_true, probs)
except Exception:
    auc = float('nan')

print("\n=== Test Metrics ===")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC AUC  : {auc:.4f}")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:\n", cm)

# Save classification report
report = classification_report(y_true, y_pred, target_names=list(test_generator.class_indices.keys()), zero_division=0)
with open("classification_report.txt", "w") as f:
    f.write("=== Test Metrics ===\n")
    f.write(f"Accuracy : {acc:.4f}\nPrecision: {prec:.4f}\nRecall   : {rec:.4f}\nF1-score : {f1:.4f}\nROC AUC  : {auc:.4f}\n\n")
    f.write("=== Classification Report ===\n")
    f.write(report)

# ==== PLOTS ====
# Training curves
plt.figure()
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history.get('val_loss', []), label='val_loss')
plt.xlabel('Epoch'); plt.ylabel('Binary Crossentropy'); plt.title('Training Loss'); plt.legend()
plt.tight_layout(); plt.savefig('training_loss.png', dpi=200)

plt.figure()
plt.plot(history.history.get('accuracy', []), label='train_acc')
plt.plot(history.history.get('val_accuracy', []), label='val_acc')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Training Accuracy'); plt.legend()
plt.tight_layout(); plt.savefig('training_accuracy.png', dpi=200)

plt.figure()
plt.plot(history.history.get('auc', []), label='train_auc')
plt.plot(history.history.get('val_auc', []), label='val_auc')
plt.xlabel('Epoch'); plt.ylabel('ROC AUC'); plt.title('Training AUC'); plt.legend()
plt.tight_layout(); plt.savefig('training_auc.png', dpi=200)

# Confusion matrix plot
plt.figure()
import itertools
classes = list(test_generator.class_indices.keys())
plt.imshow(cm, interpolation='nearest')
plt.title('Confusion Matrix'); plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45); plt.yticks(tick_marks, classes)
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
plt.ylabel('True label'); plt.xlabel('Predicted label')
plt.tight_layout(); plt.savefig('confusion_matrix.png', dpi=200)

# ==== SAVE FINAL MODEL ====
model.save("stego_cnn_model.keras")
print("\nTraining complete. Best model saved as best_model.keras; final snapshot saved as stego_cnn_model.keras")
print("Artifacts: training_loss.png, training_accuracy.png, training_auc.png, confusion_matrix.png, classification_report.txt")
