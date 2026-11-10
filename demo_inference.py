import os
import sys
import glob
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

IMAGE_SIZE = (32, 32)
CLASS_NAMES = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]
NUM_CLASSES = 10

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
    return model

# 1. Nạp mô hình & Trọng số đã huấn luyện
model = build_fixed_vgg16()
weight_file = "real_superior_best_model.weights.h5"
if os.path.exists(weight_file):
    model.load_weights(weight_file)
    print(f"✅ Đã nạp weights thành công từ: {weight_file}")
else:
    print(f"❌ Không tìm thấy file {weight_file}!")
    sys.exit(1)

# 2. Lấy mẫu ngẫu nhiên từ mỗi lớp của tập kiểm chuẩn độc lập (superior_clean_label_book)
CLEAN_LB_DIR = "superior_clean_label_book"
test_samples = []

for c in CLASS_NAMES:
    c_files = glob.glob(f"{CLEAN_LB_DIR}/{c}/*.png")
    if c_files:
        # Chọn 1-2 ảnh đại diện mỗi lớp
        selected = random.sample(c_files, min(1, len(c_files)))
        for s in selected:
            test_samples.append((s, c))

# Thêm 2 ảnh bất kỳ để đủ 12 ảnh (lưới 3x4)
all_lb = glob.glob(f"{CLEAN_LB_DIR}/*/*.png")
while len(test_samples) < 12:
    pick = random.choice(all_lb)
    cname = os.path.basename(os.path.dirname(pick))
    if (pick, cname) not in test_samples:
        test_samples.append((pick, cname))

test_samples = test_samples[:12]

# 3. Chạy dự đoán thực tế
print("\n" + "="*75)
print("🔍 KẾT QUẢ DỰ ĐOÁN THỰC TẾ TRÊN TỪNG ẢNH TEST ĐỘC LẬP:")
print("="*75)

images_to_plot = []
results_info = []

for idx, (img_path, true_label) in enumerate(test_samples, 1):
    with Image.open(img_path) as im:
        img_rgb = im.convert("RGB")
        arr = np.array(img_rgb, dtype=np.float32)
        
    # Dự đoán
    inp = np.expand_dims(arr, axis=0)
    preds = model.predict(inp, verbose=0)[0]
    pred_idx = np.argmax(preds)
    pred_label = CLASS_NAMES[pred_idx]
    confidence = preds[pred_idx] * 100
    
    is_correct = (pred_label == true_label)
    status_icon = "✅ CHÍNH XÁC" if is_correct else "❌ NHẦM LẪN"
    
    filename = os.path.basename(img_path)
    print(f"[{idx:02d}] File: {filename:<36} | Thực tế: {true_label.upper():<4} ➔ Đoán: {pred_label.upper():<4} ({confidence:5.1f}%) | {status_icon}")
    
    images_to_plot.append(img_rgb)
    results_info.append({
        "true": true_label,
        "pred": pred_label,
        "conf": confidence,
        "correct": is_correct
    })

print("="*75)

# 4. Vẽ ảnh minh họa trực quan 12 mẫu (lưới 3 x 4)
fig, axes = plt.subplots(3, 4, figsize=(14, 11))
axes = axes.flatten()

for i, ax in enumerate(axes):
    img = images_to_plot[i]
    info = results_info[i]
    
    # Phóng to ảnh để nhìn rõ nét chữ 32x32
    ax.imshow(img)
    ax.axis("off")
    
    color = "#27ae60" if info["correct"] else "#e74c3c"
    status_text = "CHÍNH XÁC" if info["correct"] else "SAI"
    
    title = f"Thực tế: {info['true'].upper()}  ➔  Đoán: {info['pred'].upper()}\nĐộ tin cậy: {info['conf']:.1f}% ({status_text})"
    ax.set_title(title, fontsize=11, fontweight="bold", color=color, pad=8)

plt.suptitle("🏛️ THỰC NGHIỆM TRỰC QUAN: MÔ HÌNH VGG16 NHẬN DIỆN CHỮ SỐ LA MÃ", fontsize=15, fontweight="bold", y=0.98)
plt.tight_layout()
output_img = "demo_predictions_showcase.png"
plt.savefig(output_img, dpi=300)
plt.close()
print(f"\n🖼️ Đã xuất ảnh minh họa trực quan đẹp mắt: {output_img}")
