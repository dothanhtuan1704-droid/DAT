# 🏛️ DAT301m - Data-Centric AI Competition (Handwritten Roman Numerals)
## Tối Ưu Hóa Dữ Liệu Nhận Dạng Chữ Số La Mã Với Mạng VGG-16 Scratch Cố Định

Repo chứa toàn bộ sản phẩm thực nghiệm của nhóm (bao gồm bộ 6 notebook độc lập từ Stage 0 đến Stage 5 có sẵn kết quả chạy Google Colab và file ZIP đóng gói).

---

### 📦 Tải Nhanh File ZIP Đóng Gói (Chưa Gộp)
👉 **[DAT301m_Separate_Stages_Notebooks.zip](DAT301m_Separate_Stages_Notebooks.zip)** *(Chứa đầy đủ 6 Notebook độc lập, Báo cáo sai nhãn và Hình ảnh minh chứng)*

---

### 📊 Bảng Tiến Trình Thực Nghiệm Qua 5 Stage (15 Epochs, VGG-16 Scratch, Colab T4 GPU):

| Notebook | Tên Stage | Quy Mô Dữ Liệu | Val Accuracy | Label Book (Test) | Điểm Đột Phá Khoa Học |
| :--- | :--- | :---: | :---: | :---: | :--- |
| [00_Stage0_Baseline.ipynb](00_Stage0_Baseline.ipynb) | Stage 0: Baseline | 2.880 ảnh thô | **65.81%** *(Điểm ảo)* | **57.69%** | Phát hiện 49 ảnh rò rỉ (MD5) & ảnh trắng 100% |
| [01_Stage1_Data_Cleaning.ipynb](01_Stage1_Data_Cleaning.ipynb) | Stage 1: Data Cleaning | 2.830 ảnh sạch | **60.88%** *(Điểm thật)* | **55.77%** | Khử sạch rò rỉ, loại bỏ ảnh trắng & mẫu gán sai nhãn |
| [02_Stage2_Geometric_Normalization.ipynb](02_Stage2_Geometric_Normalization.ipynb) | Stage 2: Geometric Normalization | 2.830 ảnh căn giữa | **81.42%** *(+20.5%)* | **84.62%** *(+28.9%)* | **Bước nhảy vọt:** Căn giữa Bounding Box & tẩy trắng nền scan |
| [03_Stage3_Balanced_Augmentation.ipynb](03_Stage3_Balanced_Augmentation.ipynb) | Stage 3: Balanced Augmentation | 7.500 ảnh (750/lớp) | **90.20%** | **80.77%** | Vượt mốc 90% nhờ cân bằng lớp, cấm lật ngang |
| [04_Stage4_Max_Capacity_Optimization.ipynb](04_Stage4_Max_Capacity_Optimization.ipynb) | Stage 4: Max Capacity Optimization | 9.500 ảnh (950/lớp) | **91.21%** *(Đỉnh cao)* | **80.77%** | Hội tụ tốt nhất ở Epoch 13, tối đa hóa trần 10.000 ảnh |
| [05_Stage5_Final_Evaluation_and_Comparison.ipynb](05_Stage5_Final_Evaluation_and_Comparison.ipynb) | Stage 5: Final Evaluation & Benchmark | Nạp weights Stage 4 | — | **80.77%** | Xuất Confusion Matrix 10x10 & Biểu đồ Ablation Study |

---

### 🔍 Danh Sách Ảnh Sai Nhãn Được Phát Hiện:
Xem chi tiết tại: [danh_sach_chi_tiet_anh_sai_label.md](danh_sach_chi_tiet_anh_sai_label.md)
* Ảnh trắng xóa 100%: `data/train/vi/aac26b40-ce5d-11eb-b317-38f9d35ea60f.png`
* Chữ `X` bị gán nhãn `iii`: `data/train/iii/b0a13faa-ce5d-11eb-b317-38f9d35ea60f.png`
* Chữ `I` bị gán nhãn `iii`: `data/train/iii/b030ba5a-ce5d-11eb-b317-38f9d35ea60f.png`
* Chữ `VI` bị gán nhãn `ix`: `data/train/ix/aec401ea-ce5d-11eb-b317-38f9d35ea60f.png`