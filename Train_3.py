import os, glob, random, math, itertools
import numpy as np
from PIL import Image
import tensorflow as tf
from keras import ops, layers, models, callbacks, regularizers  # Keras 3 imports
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt

# ========= CONFIG =========
DATASET_DIR   = "dataset"
SEED          = 42
IMG_PATCH     = 256           # patch side
BATCH_SIZE    = 32
EPOCHS        = 50
BASE_LR       = 1e-3
WEIGHT_DECAY  = 1e-4
VAL_SPLIT     = 0.15
USE_FLIPS     = True
NUM_WORKERS   = tf.data.AUTOTUNE

os.environ["TF_DETERMINISTIC_OPS"] = "1"
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# ========= FILE LISTS =========
def list_files(root):
    cover = sorted(glob.glob(os.path.join(root, "cover", "*")))
    stego = sorted(glob.glob(os.path.join(root, "stego", "*")))
    return cover, stego

train_cover, train_stego = list_files(os.path.join(DATASET_DIR, "train"))
test_cover,  test_stego  = list_files(os.path.join(DATASET_DIR, "test"))

def split_stratified(covers, stegos, val_ratio):
    n_c, n_s = len(covers), len(stegos)
    n_val_c  = int(round(n_c * val_ratio))
    n_val_s  = int(round(n_s * val_ratio))
    rng = np.random.default_rng(SEED)
    val_idx_c = set(rng.choice(n_c, n_val_c, replace=False))
    val_idx_s = set(rng.choice(n_s, n_val_s, replace=False))
    tr_c = [p for i,p in enumerate(covers) if i not in val_idx_c]
    vl_c = [p for i,p in enumerate(covers) if i in val_idx_c]
    tr_s = [p for i,p in enumerate(stegos) if i not in val_idx_s]
    vl_s = [p for i,p in enumerate(stegos) if i in val_idx_s]
    return (tr_c, tr_s), (vl_c, vl_s)

(train_cover, train_stego), (val_cover, val_stego) = split_stratified(train_cover, train_stego, VAL_SPLIT)
print(f"Train: cover={len(train_cover)} stego={len(train_stego)} | "
      f"Val: cover={len(val_cover)} stego={len(val_stego)} | "
      f"Test: cover={len(test_cover)} stego={len(test_stego)}")

# ========= IMAGE IO / PATCHING =========
def load_gray(path):
    with Image.open(path) as im:
        im = im.convert("L")
        arr = np.array(im, dtype=np.uint8)
    return arr

def center_crop(arr, size):
    h, w = arr.shape
    ch = min(h, size); cw = min(w, size)
    y0 = (h - ch)//2; x0 = (w - cw)//2
    crop = arr[y0:y0+ch, x0:x0+cw]
    pad_h = size - ch; pad_w = size - cw
    if pad_h>0 or pad_w>0:
        crop = np.pad(crop, ((pad_h//2, pad_h - pad_h//2),
                             (pad_w//2, pad_w - pad_w//2)), mode='edge')
    return crop

def random_crop(arr, size, rng):
    h, w = arr.shape
    if h < size or w < size:
        return center_crop(arr, size)
    y0 = rng.integers(0, h - size + 1); x0 = rng.integers(0, w - size + 1)
    return arr[y0:y0+size, x0:x0+size]

def tf_augment(img):
    if USE_FLIPS:
        img = tf.image.random_flip_left_right(img, seed=SEED)
        img = tf.image.random_flip_up_down(img, seed=SEED)
    return img

rng_np = np.random.default_rng(SEED)

def make_list_with_labels(covers, stegos):
    X = [(p, 0) for p in covers] + [(p, 1) for p in stegos]
    rng_np.shuffle(X)
    return X

train_list = make_list_with_labels(train_cover, train_stego)
val_list   = make_list_with_labels(val_cover,   val_stego)
test_list  = make_list_with_labels(test_cover,  test_stego)

def gen_batch(file_label_list, training):
    rng = np.random.default_rng(SEED)
    while True:
        rng.shuffle(file_label_list)
        for i in range(0, len(file_label_list), BATCH_SIZE):
            batch = file_label_list[i:i+BATCH_SIZE]
            imgs, labels = [], []
            for path, lab in batch:
                arr = load_gray(path)
                arr = random_crop(arr, IMG_PATCH, rng) if training else center_crop(arr, IMG_PATCH)
                arr = arr.astype(np.float32) / 255.0
                imgs.append(arr[..., None])
                labels.append(lab)
            x = np.stack(imgs, 0)
            y = np.array(labels, dtype=np.float32)
            yield x, y

steps_tr = math.ceil(len(train_list) / BATCH_SIZE)
steps_vl = math.ceil(len(val_list)   / BATCH_SIZE)
steps_te = math.ceil(len(test_list)  / BATCH_SIZE)

train_ds = tf.data.Dataset.from_generator(
    lambda: gen_batch(train_list, True),
    output_signature=(
        tf.TensorSpec(shape=(None, IMG_PATCH, IMG_PATCH, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
    )
).map(lambda x,y: (tf_augment(x), y), num_parallel_calls=NUM_WORKERS).prefetch(NUM_WORKERS)

val_ds = tf.data.Dataset.from_generator(
    lambda: gen_batch(val_list, False),
    output_signature=(
        tf.TensorSpec(shape=(None, IMG_PATCH, IMG_PATCH, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
    )
).prefetch(NUM_WORKERS)

test_ds = tf.data.Dataset.from_generator(
    lambda: gen_batch(test_list, False),
    output_signature=(
        tf.TensorSpec(shape=(None, IMG_PATCH, IMG_PATCH, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
    )
)

# ========= MODEL: Bayar + TLU + ResNet+SE =========
class BayarConstraint(tf.keras.constraints.Constraint):
    """Constrain each kernel so sum of weights = 0 and center weight = -sum(others)."""
    def __init__(self, kh, kw):
        self.kh, self.kw = kh, kw
        self.center = (kh//2, kw//2)
    def __call__(self, w):
        kh, kw = self.kh, self.kw
        cy, cx = self.center
        mask_center = tf.one_hot(cy, kh)[:,None]*tf.one_hot(cx, kw)[None,:]
        mask_center = tf.reshape(mask_center, (kh,kw,1,1))
        mask_center = tf.cast(mask_center, w.dtype)
        mask_others = 1.0 - mask_center
        w_others = w * mask_others
        sum_others = tf.reduce_sum(w_others, axis=[0,1], keepdims=True)
        w_center = -sum_others
        w = w_others + w_center * mask_center
        return w

class TLU(layers.Layer):
    """Truncated Linear Unit: clamp to [-th, +th]."""
    def __init__(self, th=3.0, **kw):
        super().__init__(**kw); self.th = float(th)
    def call(self, x):
        return ops.clip(x, -self.th, self.th)
    def get_config(self):
        return {"th": self.th, **super().get_config()}

def se_block(x, ratio=16):
    ch = x.shape[-1]
    s = layers.GlobalAveragePooling2D()(x)
    s = layers.Dense(ch//ratio, activation='relu')(s)
    s = layers.Dense(ch, activation='sigmoid')(s)
    s = layers.Reshape((1,1,ch))(s)
    return layers.Multiply()([x, s])

def res_block(x, filters, wd=WEIGHT_DECAY):
    y = layers.Conv2D(filters, 3, padding='same',
                      kernel_regularizer=regularizers.l2(wd), use_bias=False)(x)
    y = layers.BatchNormalization()(y); y = layers.ReLU()(y)
    y = layers.Conv2D(filters, 3, padding='same',
                      kernel_regularizer=regularizers.l2(wd), use_bias=False)(y)
    y = layers.BatchNormalization()(y)
    y = se_block(y)
    y = layers.Add()([x, y])
    y = layers.ReLU()(y)
    return y

def down_block(x, filters, wd=WEIGHT_DECAY):
    x = layers.AveragePooling2D(2)(x)
    x = layers.Conv2D(filters, 1, padding='same',
                      kernel_regularizer=regularizers.l2(wd), use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    return x

def build_model():
    inp = layers.Input((IMG_PATCH, IMG_PATCH, 1))
    # Bayar front-end
    bayar = layers.Conv2D(
        filters=32, kernel_size=5, padding='same', use_bias=False,
        kernel_constraint=BayarConstraint(5,5),
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
        name="bayar_conv")(inp)
    # TLU clamp (Keras 3-safe)
    x = TLU(3.0, name="tlu")(bayar)

    # Stem
    x = layers.Conv2D(64, 3, padding='same',
                      kernel_regularizer=regularizers.l2(WEIGHT_DECAY), use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)

    # Residual stages
    for _ in range(3): x = res_block(x, 64)
    x = down_block(x, 96)
    for _ in range(3): x = res_block(x, 96)
    x = down_block(x, 128)
    for _ in range(4): x = res_block(x, 128)
    x = down_block(x, 160)
    for _ in range(4): x = res_block(x, 160)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation='sigmoid', name="p_stego")(x)
    return models.Model(inp, out, name="BayarTLU_ResSE")

model = build_model()

# Cosine LR schedule
cosine = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=BASE_LR, first_decay_steps=math.ceil( len(train_list)/BATCH_SIZE )*8,
    t_mul=2.0, m_mul=0.8, alpha=1e-2
)
opt = tf.keras.optimizers.Adam(learning_rate=cosine)

model.compile(
    optimizer=opt,
    loss='binary_crossentropy',
    metrics=[tf.keras.metrics.AUC(name='auc'), 'accuracy']
)
model.summary()

def class_weights_from_list(L):
    y = np.array([lab for _,lab in L], dtype=np.int32)
    n = len(y); pos = y.sum(); neg = n - pos
    return {0: n/(2*neg) if neg else 1.0, 1: n/(2*pos) if pos else 1.0}
class_weights = class_weights_from_list(train_list)

ckp = callbacks.ModelCheckpoint("best_model.keras", monitor='val_auc', mode='max', save_best_only=True)
es  = callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=8, restore_best_weights=True)
rlr = callbacks.ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=4, min_lr=1e-6)

history = model.fit(
    train_ds,
    steps_per_epoch=math.ceil(len(train_list)/BATCH_SIZE),
    validation_data=val_ds,
    validation_steps=math.ceil(len(val_list)/BATCH_SIZE),
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[ckp, es, rlr],
    verbose=1
)

# ====== EVALUATION ======
def collect_probs(ds):
    probs, labels = [], []
    for x, y in ds:
        p = model.predict(x, verbose=0).ravel()
        probs.append(p); labels.append(y)
    return np.concatenate(probs), np.concatenate(labels).astype(int)

val_probs, y_val   = collect_probs(val_ds)
test_probs, y_test = collect_probs(test_ds)

# Tune thresholds on validation
ths = np.linspace(0.01, 0.99, 99)
best_f1, thr_f1 = -1, 0.5
best_j,  thr_j  = -1, 0.5
for t in ths:
    yv = (val_probs >= t).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_val, yv, average='binary', zero_division=0)
    cm = confusion_matrix(y_val, yv)
    tn, fp, fn, tp = cm.ravel()
    tpr = tp/(tp+fn) if (tp+fn) else 0.; tnr = tn/(tn+fp) if (tn+fp) else 0.
    j = tpr + tnr - 1
    if f1 > best_f1: best_f1, thr_f1 = f1, t
    if j  > best_j:  best_j,  thr_j  = j,  t

def eval_at(probs, y_true, thr, name):
    y_pred = (probs >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    try: auc = roc_auc_score(y_true, probs)
    except: auc = float('nan')
    cm = confusion_matrix(y_true, y_pred)
    rep = classification_report(y_true, y_pred, target_names=["cover","stego"], zero_division=0)
    print(f"\n=== TEST @ {name} (thr={thr:.2f}) ===")
    print(f"Acc {acc:.4f} | Prec {prec:.4f} | Rec {rec:.4f} | F1 {f1:.4f} | AUC {auc:.4f}")
    print("CM:\n", cm)
    with open(f"classification_report_thr_{name.replace(' ','_')}.txt","w") as f:
        f.write(rep)
    return cm

cm0  = eval_at(test_probs, y_test, 0.5, "0.5_default")
cmJ  = eval_at(test_probs, y_test, thr_j, "YoudenJ")
cmF1 = eval_at(test_probs, y_test, thr_f1, "BestF1")

# ====== PLOTS ======
plt.figure()
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history.get('val_loss', []), label='val_loss')
plt.legend(); plt.xlabel('Epoch'); plt.ylabel('BCE'); plt.title('Training Loss')
plt.tight_layout(); plt.savefig('training_loss.png', dpi=200)

plt.figure()
plt.plot(history.history.get('auc', []), label='train_auc')
plt.plot(history.history.get('val_auc', []), label='val_auc')
plt.legend(); plt.xlabel('Epoch'); plt.ylabel('AUC'); plt.title('Training AUC')
plt.tight_layout(); plt.savefig('training_auc.png', dpi=200)

plt.figure()
plt.plot(history.history.get('accuracy', []), label='train_acc')
plt.plot(history.history.get('val_accuracy', []), label='val_acc')
plt.legend(); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Training Accuracy')
plt.tight_layout(); plt.savefig('training_acc.png', dpi=200)

plt.figure()
plt.hist(test_probs[y_test==0], bins=40, alpha=0.6, label='cover')
plt.hist(test_probs[y_test==1], bins=40, alpha=0.6, label='stego')
plt.legend(); plt.xlabel('p(stego)'); plt.ylabel('count'); plt.title('Test score distribution')
plt.tight_layout(); plt.savefig('score_histogram.png', dpi=200)

plt.figure()
plt.imshow(cm0, interpolation='nearest')
plt.title('Confusion Matrix @ 0.5'); plt.colorbar()
ticks = np.arange(2)
plt.xticks(ticks, ['cover','stego'], rotation=45); plt.yticks(ticks, ['cover','stego'])
m = cm0.max()/2
for i,j in itertools.product(range(2), range(2)):
    plt.text(j, i, format(cm0[i,j], 'd'),
             horizontalalignment='center',
             color='white' if cm0[i,j] > m else 'black')
plt.ylabel('True'); plt.xlabel('Predicted')
plt.tight_layout(); plt.savefig('confusion_matrix.png', dpi=200)

model.save("final_model.keras")
print("\nSaved: best_model.keras, final_model.keras and all plots/reports.")
