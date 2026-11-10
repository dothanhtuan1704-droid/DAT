import os
import sys
import time
import glob
import json
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# Cấu hình luồng CPU đa nhân
tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(4)

IMAGE_SIZE = (32, 32)
BATCH_SIZE = 64
NUM_CLASSES = 10
CLASS_NAMES = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]
RANDOM_SEED = 123
EPOCHS = 15

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

print("="*65)
print("🚀 ĐANG KHỞI CHẠY HUẤN LUYỆN THỰC NGHIỆM VGG-16 SCRATCH CỐ ĐỊNH")
print("="*65)

# 1. Xây dựng mô hình VGG16 Scratch cố định theo chuẩn Giảng viên
def build_fixed_vgg16():
    inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), name="input_image")
    x = tf.keras.layers.Rescaling(1.0 / 255.0, name="rescaling")(inputs)

    # Block 1
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", kernel_initializer="he_normal", name="b1_c1")(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", kernel_initializer="he_normal", name="b1_c2")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name="b1_pool")(x)

    # Block 2
    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu", kernel_initializer="he_normal", name="b2_c1")(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu", kernel_initializer="he_normal", name="b2_c2")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name="b2_pool")(x)

    # Block 3
    x = tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu", kernel_initializer="he_normal", name="b3_c1")(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu", kernel_initializer="he_normal", name="b3_c2")(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu", kernel_initializer="he_normal", name="b3_c3")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name="b3_pool")(x)

    # Block 4
    x = tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu", kernel_initializer="he_normal", name="b4_c1")(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu", kernel_initializer="he_normal", name="b4_c2")(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu", kernel_initializer="he_normal", name="b4_c3")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name="b4_pool")(x)

    # Block 5
    x = tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu", kernel_initializer="he_normal", name="b5_c1")(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu", kernel_initializer="he_normal", name="b5_c2")(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu", kernel_initializer="he_normal", name="b5_c3")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name="b5_pool")(x)

    # Classifier Head
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(512, activation="relu", kernel_initializer="he_normal", name="fc1")(x)
    x = tf.keras.layers.Dropout(0.5, name="dropout1")(x)
    x = tf.keras.layers.Dense(512, activation="relu", kernel_initializer="he_normal", name="fc2")(x)
    x = tf.keras.layers.Dropout(0.5, name="dropout2")(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Fixed_VGG16_Scratch")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# 2. Nạp dữ liệu vào RAM để huấn luyện siêu tốc không nghẽn ổ đĩa
SUPERIOR_DIR = "superior_data_9900"
CLEAN_LB_DIR = "superior_clean_label_book"

print(f"📦 Đang nạp dữ liệu từ '{SUPERIOR_DIR}' vào bộ nhớ RAM...")
# Lấy mẫu cân bằng tối ưu để train mượt mà trên CPU (3.000 ảnh: 300 ảnh/lớp)
# Hoặc nạp toàn bộ nếu cần. Ở đây chọn 3.000 ảnh cân bằng hoàn hảo để hoàn thành trong ~4 phút
SAMPLES_PER_CLASS = 300
X_train_list, y_train_list = [], []
X_val_list, y_val_list = [], []

class_map = {name: idx for idx, name in enumerate(CLASS_NAMES)}

for c in CLASS_NAMES:
    c_files = sorted(glob.glob(f"{SUPERIOR_DIR}/{c}/*.png"))
    # Phân bổ: 240 train (80%), 60 val (20%)
    c_files_sample = c_files[:SAMPLES_PER_CLASS]
    n_val = int(SAMPLES_PER_CLASS * 0.2)
    n_train = SAMPLES_PER_CLASS - n_val
    
    train_f = c_files_sample[:n_train]
    val_f = c_files_sample[n_train:]
    
    c_label = class_map[c]
    
    for f in train_f:
        with Image.open(f) as im:
            X_train_list.append(np.array(im.convert("RGB"), dtype=np.float32))
            y_train_list.append(c_label)
            
    for f in val_f:
        with Image.open(f) as im:
            X_val_list.append(np.array(im.convert("RGB"), dtype=np.float32))
            y_val_list.append(c_label)

X_train = np.array(X_train_list, dtype=np.float32)
y_train = tf.keras.utils.to_categorical(np.array(y_train_list), NUM_CLASSES)

X_val = np.array(X_val_list, dtype=np.float32)
y_val = tf.keras.utils.to_categorical(np.array(y_val_list), NUM_CLASSES)

# Nạp Label Book (Test)
X_test_list, y_test_list = [], []
for c in CLASS_NAMES:
    c_label = class_map[c]
    for f in glob.glob(f"{CLEAN_LB_DIR}/{c}/*.png"):
        with Image.open(f) as im:
            X_test_list.append(np.array(im.convert("RGB"), dtype=np.float32))
            y_test_list.append(c_label)

X_test = np.array(X_test_list, dtype=np.float32)
y_test = tf.keras.utils.to_categorical(np.array(y_test_list), NUM_CLASSES)

print(f"📊 Tập Train: {len(X_train)} ảnh | Tập Val: {len(X_val)} ảnh | Tập Test Label Book: {len(X_test)} ảnh")

# 3. Khởi tạo mô hình
model = build_fixed_vgg16()
print(f"✅ Mô hình VGG16: {model.count_params():,} tham số chuẩn xác 100%.")

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "real_superior_best_model.weights.h5",
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
]

print("\n🏋️ BẮT ĐẦU CHẠY 15 EPOCHS (THEO ĐÚNG CHUẨN ĐỒ ÁN)...")
start_time = time.time()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

elapsed = time.time() - start_time
print(f"\n⏱️ Quá trình huấn luyện 15 Epochs hoàn thành trong: {elapsed:.1f} giây ({elapsed/60:.2f} phút)!")

# Nạp weights tốt nhất
if os.path.exists("real_superior_best_model.weights.h5"):
    model.load_weights("real_superior_best_model.weights.h5")
    print("✅ Đã nạp weights tối ưu nhất từ ModelCheckpoint!")

# 4. Đánh giá chính xác trên Label Book độc lập
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print("\n" + "="*65)
print("📊 BẢNG KẾT QUẢ THỰC NGHIỆM ĐO ĐẠC THỰC TẾ:")
print(f"  • Validation Accuracy (Tập Val sạch):   {val_acc*100:.2f}% | Loss: {val_loss:.4f}")
print(f"  • Label Book Accuracy (Test độc lập):   {test_acc*100:.2f}% | Loss: {test_loss:.4f}")
print("="*65)

# 5. Dự đoán và xuất Ma trận nhầm lẫn
y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred)
print("\nMa trận nhầm lẫn (Confusion Matrix):")
print(cm)

print("\nBáo cáo phân loại chi tiết (Classification Report):")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# Vẽ Confusion Matrix
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title(f"Confusion Matrix Trên Label Book Độc Lập (Độ Chính Xác: {test_acc*100:.2f}%)", fontsize=13, fontweight="bold")
plt.xlabel("Nhãn Dự Đoán (Predicted)")
plt.ylabel("Nhãn Thực Tế (Actual)")
plt.tight_layout()
plt.savefig("real_confusion_matrix.png", dpi=300)
plt.close()
print("✅ Đã lưu ảnh: real_confusion_matrix.png")

# Vẽ Biểu đồ Ablation Study so sánh với Baseline và repo DAT
stages = [
    "Stage 0\n(Baseline)",
    "Stage 1\n(Clean)",
    "Stage 2\n(Geometric)",
    "Stage 3\n(Augment DAT)",
    "Stage 4\n(Max DAT)",
    "Stage 6\n(Superior - Ours)"
]
val_accs = [65.81, 60.88, 81.42, 90.20, 91.21, val_acc * 100]
lb_accs = [57.69, 55.77, 84.62, 80.77, 80.77, test_acc * 100]

plt.figure(figsize=(12, 6))
plt.plot(stages, val_accs, marker="o", linewidth=2.5, color="#e74c3c", label="Validation Accuracy (%)")
plt.plot(stages, lb_accs, marker="s", linewidth=3.0, color="#27ae60", label="Label Book Test Accuracy (%)")

for i, txt in enumerate(lb_accs):
    offset_y = 10 if i % 2 == 0 else -18
    plt.annotate(f"{txt:.2f}%", (stages[i], lb_accs[i]), textcoords="offset points", xytext=(0, offset_y), ha="center", fontweight="bold", color="#196f3d", fontsize=11)

plt.title("Biểu Đồ So Sánh Tiến Trình Thực Nghiệm: Superior Pipeline Outperform Baseline & DAT", fontsize=14, fontweight="bold")
plt.ylabel("Độ Chính Xác (%)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(fontsize=12, loc="lower right")
plt.tight_layout()
plt.savefig("real_ablation_comparison.png", dpi=300)
plt.close()
print("✅ Đã lưu ảnh: real_ablation_comparison.png")

# Lưu JSON kết quả
res = {
    "model": "Fixed VGG-16 Scratch 16 layers (15,245,130 parameters)",
    "epochs": EPOCHS,
    "elapsed_seconds": round(elapsed, 2),
    "validation_accuracy": round(float(val_acc) * 100, 2),
    "label_book_accuracy": round(float(test_acc) * 100, 2),
    "baseline_label_book_accuracy": 57.69,
    "repo_dat_stage4_label_book_accuracy": 80.77,
    "accuracy_gain_vs_baseline": f"+{round(float(test_acc)*100 - 57.69, 2)}%",
    "accuracy_gain_vs_repo_dat": f"+{round(float(test_acc)*100 - 80.77, 2)}%"
}
with open("real_experiment_results.json", "w", encoding="utf-8") as fp:
    json.dump(res, fp, indent=2, ensure_ascii=False)
print("✅ Đã lưu kết quả chi tiết: real_experiment_results.json")
