# 🔍 DANH SÁCH CHI TIẾT CÁC ẢNH BỊ ĐÁNH SAI LABEL VÀ DỊ TẬT DỮ LIỆU
## Môn học: DAT301m - Cuộc thi Data-Centric AI (Handwritten Roman Numerals)

---

### 📌 PHẦN 1: BẢNG TRA CỨU CÁC MẪU BỊ GÁN SAI NHÃN & DỊ TẬT (LABEL NOISE & CORRUPT)

Dưới đây là các ảnh sai nhãn nghiêm trọng và ảnh hỏng được phát hiện trong tập dữ liệu gốc `data/train/`:

| STT | Tên File Ảnh | Thư Mục / Nhãn BTC Gán | Nét Chữ Thực Tế | Hiện Tượng & Bằng Chứng Thị Giác |
| :---: | :--- | :---: | :---: | :--- |
| **1** | `aac26b40-ce5d-11eb-b317-38f9d35ea60f.png` | `data/train/vi/` (Số 6) | **ẢNH TRẮNG 100%** | Toàn bộ pixel min=255, max=255. Không hề có nét mực nào (lỗi scan tài liệu). |
| **2** | `b0a13faa-ce5d-11eb-b317-38f9d35ea60f.png` | `data/train/iii/` (Số 3) | **CHỮ 'X' (Số 10)** | Có 2 nét gạch chéo rõ rệt tạo thành chữ X, nhưng bị gán nhầm vào thư mục `iii`. |
| **3** | `b030ba5a-ce5d-11eb-b317-38f9d35ea60f.png` | `data/train/iii/` (Số 3) | **CHỮ 'I' (Số 1)** | Chỉ có đúng 1 nét gạch đứng duy nhất, nhưng bị gán vào nhãn `iii`. |
| **4** | `aec401ea-ce5d-11eb-b317-38f9d35ea60f.png` | `data/train/ix/` (Số 9) | **CHỮ 'VI' (Số 6)** | Chữ 'V' bên trái, gạch 'I' bên phải -> rõ ràng là số 6, bị gán nhầm thành số 9. |
| **5** | `aec04e56-ce5d-11eb-b317-38f9d35ea60f.png` | `data/train/ix/` (Số 9) | **CHỮ 'VI' (Số 6)** | Chữ 'V' bên trái, gạch 'I' bên phải -> bị gán nhầm vào thư mục `ix`. |
| **6** | `adf5ad04-ce5d-11eb-b317-38f9d35ea60f.png` | `data/train/iv/` (Số 4) | **CHỮ 'I' (Số 1)** | Chỉ có 1 nét gạch đứng đơn độc, bị gán nhầm thành số 4 (`iv`). |

---

### 📂 PHẦN 2: VỊ TRÍ CHÍNH XÁC CỦA TỪNG ẢNH TRÊN MÁY TÍNH CỦA BẠN

Các ảnh này hiện đang nằm trong thư mục dữ liệu gốc của đồ án tại đường dẫn:
`E:\Group_DAT\deeplearningai-data-centric-competition\data\train\`

Đường dẫn tuyệt đối đến từng file:
1. `E:\Group_DAT\deeplearningai-data-centric-competition\data\train\vi\aac26b40-ce5d-11eb-b317-38f9d35ea60f.png`
2. `E:\Group_DAT\deeplearningai-data-centric-competition\data\train\iii\b0a13faa-ce5d-11eb-b317-38f9d35ea60f.png`
3. `E:\Group_DAT\deeplearningai-data-centric-competition\data\train\iii\b030ba5a-ce5d-11eb-b317-38f9d35ea60f.png`
4. `E:\Group_DAT\deeplearningai-data-centric-competition\data\train\ix\aec401ea-ce5d-11eb-b317-38f9d35ea60f.png`
5. `E:\Group_DAT\deeplearningai-data-centric-competition\data\train\ix\aec04e56-ce5d-11eb-b317-38f9d35ea60f.png`
6. `E:\Group_DAT\deeplearningai-data-centric-competition\data\train\iv\adf5ad04-ce5d-11eb-b317-38f9d35ea60f.png`

*(Đã đính kèm ảnh tổng hợp minh chứng trực quan: `slide_label_noise_evidence.png` bên trong file ZIP)*

---

### ⚠️ PHẦN 3: HIỆN TƯỢNG RÒ RỈ DỮ LIỆU (49 CẶP ẢNH TRÙNG LẶP QUA MÃ BĂM MD5)
* Có **49 file ảnh trong tập Train** trùng khớp 100% từng byte dữ liệu với các file ảnh trong tập **Validation**.
* **Nguyên nhân:** Do BTC khi phân chia tập train/val ban đầu không kiểm tra mã băm MD5, dẫn đến việc mô hình ở Stage 0 bị "học tủ" (Data Leakage) đạt điểm ảo 65.81%.
* **Xử lý:** Ở Stage 1, nhóm đã dùng thuật toán băm MD5 để loại bỏ toàn bộ 49 ảnh rò rỉ này, giúp điểm số phản ánh thực chất độ khái quát hóa.
