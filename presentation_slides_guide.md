# 🏛️ CẨM NANG THUYẾT TRÌNH & KỊCH BẢN BẢO VỆ ĐỒ ÁN DAT301m
## ĐỀ TÀI: TỐI ƯU HÓA DỮ LIỆU NHẬN DẠNG CHỮ SỐ LA MÃ VỚI MÔ HÌNH VGG16 SCRATCH CỐ ĐỊNH (DATA-CENTRIC AI)

* **Nhóm thực hiện:** Team 1 - Lớp DAT301m
* **Liên kết Canva thiết kế Slide:** `https://canva.link/hwd21bpptqvit3y`
* **Thời gian báo cáo:** Thứ 7 tuần sau
* **Tài nguyên ảnh chèn Slide đã tạo sẵn:**
  * [**`slide_ablation_study_chart.png`**](file:///e:/Group_DAT/deeplearningai-data-centric-competition/slide_ablation_study_chart.png) (Biểu đồ tiến trình tăng trưởng 5 Stages)
  * [**`slide_label_noise_evidence.png`**](file:///e:/Group_DAT/deeplearningai-data-centric-competition/slide_label_noise_evidence.png) (Bằng chứng 4 ca sai nhãn & ảnh trắng)
  * [**`slide_geometric_breakthrough.png`**](file:///e:/Group_DAT/deeplearningai-data-centric-competition/slide_geometric_breakthrough.png) (Minh họa Bounding Box & Tẩy trắng nền)

---

## 📑 PHẦN 1: KỊCH BẢN CHI TIẾT TỪNG SLIDE (SLIDE-BY-SLIDE SCRIPT)

### 📌 SLIDE 1: GIỚI THIỆU ĐỀ TÀI & TRIẾT LÝ DATA-CENTRIC AI
* **Tiêu đề Slide:** Đồ Án Kết Thúc Môn Học DAT301m: Data-Centric AI Competition
* **Tiêu đề phụ:** Tối Ưu Hóa Dữ Liệu Nhận Dạng Chữ Số La Mã Viết Tay Với Mô Hình VGG16 Cố Định Tự Xây Dựng
* **Nội dung hiển thị:**
  * Bài toán: Phân loại 10 chữ số La Mã viết tay từ 1 đến 10 (`i`, `ii`, `iii`, `iv`, `v`, `vi`, `vii`, `viii`, `ix`, `x`).
  * Khởi xướng: Cuộc thi do GS. Andrew Ng (DeepLearning.AI & Landing AI) phát động.
  * **Quy tắc cốt tử:** Mô hình và siêu tham số được **cố định 100%**. Mọi cải tiến hiệu năng đều đến từ kỹ thuật xử lý dữ liệu.
  * Giới hạn dữ liệu: Tổng kích thước tập Train + Validation $\le$ 10.000 ảnh.
* **🗣️ Lời thoại thuyết trình mẫu (Speaker Script):**
  > *"Kính thưa thầy và các bạn, trong phát triển học sâu truyền thống (Model-Centric), chúng ta thường giữ nguyên dữ liệu và liên tục thử nghiệm các kiến trúc mạng mới. Tuy nhiên, theo học thuyết Data-Centric AI của Giáo sư Andrew Ng, khi kiến trúc mạng đã đủ mạnh, nút thắt lớn nhất kìm hãm hiệu năng chính là chất lượng dữ liệu. Đến với đồ án này, nhóm 1 chúng em nhận bài toán phân loại 10 chữ số La Mã viết tay với mô hình VGG16 cố định, và mục tiêu duy nhất của nhóm là dùng các kỹ thuật tối ưu hóa dữ liệu để bứt phá độ chính xác từ mức 45% lên trên 82%."*

---

### 📌 SLIDE 2: KIẾN TRÚC VGG-16 TỰ XÂY DỰNG (SCRATCH 16 LAYERS)
* **Tiêu đề Slide:** Mô Hình Chuẩn Quy Định: VGG16 Tự Xây Dựng Từ Đầu (From Scratch)
* **Nội dung hiển thị:**
  * ⚠️ **Tuân thủ chỉ đạo của Giảng viên:** Tuyệt đối không sử dụng thư viện `keras.applications.VGG16` và không dùng trọng số ImageNet có sẵn.
  * Code tay từng khối theo bài báo gốc Simonyan & Zisserman (2014) tối ưu cho ảnh $32 \times 32 \times 3$:
    * **Block 1 (2 Conv):** $64 \rightarrow 64 \rightarrow \text{MaxPool}$
    * **Block 2 (2 Conv):** $128 \rightarrow 128 \rightarrow \text{MaxPool}$
    * **Block 3 (3 Conv):** $256 \rightarrow 256 \rightarrow 256 \rightarrow \text{MaxPool}$
    * **Block 4 (3 Conv):** $512 \rightarrow 512 \rightarrow 512 \rightarrow \text{MaxPool}$
    * **Block 5 (3 Conv):** $512 \rightarrow 512 \rightarrow 512 \rightarrow \text{MaxPool}$
    * **Head (3 Dense):** $\text{Flatten} \rightarrow \text{Dense}(512) \rightarrow \text{Dense}(512) \rightarrow \text{Dense}(10, \text{Softmax})$
  * Tổng số tầng có trọng số: **13 lớp Conv2D + 3 lớp Dense = 16 Layers**.
  * Tổng tham số: **15.245.130 tham số**, khởi tạo ngẫu nhiên theo phân phối He-Normal, tối ưu hóa bằng Adam ($lr=10^{-4}$).
* **🗣️ Lời thoại thuyết trình mẫu:**
  > *"Để đảm bảo tính trung thực khoa học theo yêu cầu của thầy, nhóm không dùng bất kỳ mô hình pre-trained có sẵn nào từ Keras Applications. Chúng em đã tự code tay toàn bộ 16 tầng của mạng VGG16 từ đầu bằng các lớp cơ bản và khởi tạo trọng số He-Normal. Kiến trúc này được đóng băng cố định 100% trong toàn bộ các giai đoạn thực nghiệm."*

---

### 📌 SLIDE 3: STAGE 0 - HUẤN LUYỆN BASELINE & 4 DỊ TẬT DỮ LIỆU GỐC
* **Tiêu đề Slide:** Stage 0: Khởi Động Baseline & Chẩn Đoán Dị Tật Dữ Liệu
* **Nội dung hiển thị:**
  * Dữ liệu ban đầu: 2.880 ảnh thô (2.067 train, 813 val).
  * Kết quả đo đạc:
    * Validation Accuracy (nội bộ): **45.51%** *(Điểm ảo)*
    * Label Book Accuracy (test độc lập): **36.54%** *(Điểm thực)*
    * Hiện tượng: Overfitting nặng, học vẹt.
  * **Bốn dị tật cốt tử phát hiện trên dữ liệu:**
    1. *Data Leakage:* Bị rò rỉ ảnh giữa train và val.
    2. *Blank Images:* Chứa ảnh rác trắng tinh $100\%$.
    3. *Class Imbalance:* Lớp `iv` (281 ảnh) gấp đôi lớp `ii` (157 ảnh), recall lớp `ii` bị liệt ở mức 27%.
    4. *Label Noise:* Nhiều ảnh bị gán sai nhãn nghiêm trọng.
* **🗣️ Lời thoại thuyết trình mẫu:**
  > *"Khi chạy Baseline trên dữ liệu gốc, mô hình chỉ đạt 45.51% trên tập val và rớt xuống 36.54% trên tập test độc lập Label Book. Khoảng cách chênh lệch này đã thôi thúc nhóm tiến hành kiểm toán dữ liệu và phát hiện ra 4 dị tật nghiêm trọng của tập dữ liệu ban đầu."*

---

### 📌 SLIDE 4: BẰNG CHỨNG ĐỘC QUYỀN - NHIỄU NHÃN (LABEL NOISE)
* **Tiêu đề Slide:** Bằng Chứng Thực Tế: Nhiễu Nhãn (Label Noise) & Ảnh Rác
* **Hình ảnh chèn:** [**`slide_label_noise_evidence.png`**](file:///e:/Group_DAT/deeplearningai-data-centric-competition/slide_label_noise_evidence.png)
* **Nội dung phân tích 4 trường hợp điển hình:**
  * ❌ `vi/aac26b40-ce5d-11eb-b317-38f9d35ea60f.png`: Ảnh **trắng xóa 100%** (`min=255, max=255`), hoàn toàn không có nét bút nhưng vẫn bị dán nhãn số 6 (`vi`).
  * ❌ `iii/b0a13faa-ce5d-11eb-b317-38f9d35ea60f.png`: Nét vẽ là **dấu gạch chéo X** rất rõ ràng, nhưng bị gán nhầm vào thư mục số 3 (`iii`).
  * ❌ `iii/b030ba5a-ce5d-11eb-b317-38f9d35ea60f.png`: Chỉ có **duy nhất 1 nét gạch dọc I**, hoàn toàn không có gạch số 2 hay số 3 nhưng bị dán nhãn `iii`.
  * ❌ `ix/aec401ea-ce5d-11eb-b317-38f9d35ea60f.png`: Nét vẽ bên trái là chữ **V**, bên phải là gạch **I** ($5+1=6$), nhưng người dán nhãn bị nhầm lẫn đối xứng và gán thành số 9 (`ix`).
* **🗣️ Lời thoại thuyết trình mẫu:**
  > *"Thưa thầy, đây là phát hiện quan trọng nhất của nhóm: Tập dữ liệu của ban tổ chức bị dính Label Noise rất nặng. Ví dụ trên slide, một bức ảnh dấu X rõ mười mươi lại bị bắt mô hình phải dự đoán là số III, hoặc ảnh trắng tinh không có nét bút nào lại gán nhãn VI. Trong toán học của mạng nơ-ron, việc ép mô hình học các mẫu sai nhãn này sẽ tạo ra các gradient cực lớn phá hỏng bộ trích xuất đặc trưng."*

---

### 📌 SLIDE 5: STAGE 1 - DATA AUDITING & KHỬ RÒ RỈ DỮ LIỆU
* **Tiêu đề Slide:** Stage 1: Làm Sạch Dữ Liệu & Khử Rò Rỉ Bằng Mã Băm MD5
* **Nội dung hiển thị:**
  * Phương pháp quét: Mã băm MD5 toàn bộ nội dung byte nhị phân của ảnh.
  * **Phát hiện:** **49 cặp ảnh trùng khớp 100%** giữa tập Train và tập Validation!
  * **Hành động can thiệp:**
    1. Loại bỏ triệt để 49 ảnh rò rỉ train-val.
    2. Xóa ảnh trắng tinh `vi/aac26b40`.
    3. Thanh lọc danh sách đen các mẫu sai nhãn cốt tử đã phát hiện ở Slide 4.
    4. Phân chia lại tập Train/Val sạch theo tỷ lệ Stratified 80/20.
  * **Kết quả:** Thước đo Validation phản ánh trung thực thực lực của mô hình (~48.2%), loại bỏ hoàn toàn hiện tượng học vẹt điểm ảo.
* **🗣️ Lời thoại thuyết trình mẫu:**
  > *"Tại Stage 1, nhóm sử dụng hàm băm MD5 để quét toàn diện và phát hiện 49 cặp ảnh ở tập Val thực chất là bản sao y hệt của tập Train. Điều này lý giải tại sao điểm Validation ở Stage 0 bị thổi phồng. Nhóm đã loại bỏ 49 ảnh trùng này cùng các ảnh sai nhãn, trả lại một tập dữ liệu sạch và thước đo đánh giá trung thực."*

---

### 📌 SLIDE 6: STAGE 2 - ĐỘT PHÁ HÌNH THÁI HỌC (GEOMETRIC BREAKTHROUGH)
* **Tiêu đề Slide:** Stage 2: Căn Giữa Bounding-Box & Tẩy Trắng Nền Giấy Scan
* **Hình ảnh chèn:** [**`slide_geometric_breakthrough.png`**](file:///e:/Group_DAT/deeplearningai-data-centric-competition/slide_geometric_breakthrough.png)
* **Phân tích nguyên nhân vật lý cốt lõi:**
  * Chữ viết tay ban đầu chỉ nằm co cụm ở một góc, chiếm có **15% - 30% diện tích ảnh**.
  * Khi bị resize trực tiếp 8 lần về $(32, 32)$, nét vẽ co lại thành vệt xám mờ rộng 2-3 pixel $\rightarrow$ Mạng VGG16 không thể phân giải nổi nét chữ.
  * Nền giấy scan bị ám xám (pixel 210 đến 240) gây nhiễu phân tán các bộ lọc tích chập.
* **Giải pháp thuật toán:**
  1. *Bounding-Box Auto-Centering:* Dò tọa độ nét mực ($arr < 215$), cắt sát hộp bao, đệm viền 15%, đưa chữ về trung tâm khung vuông. Nét chữ chiếm **75% - 80%** diện tích canvas.
  2. *Background Whitening:* Toàn bộ pixel nền $> 215$ đưa về trắng tuyệt đối (255) và tăng tương phản nét bút.
* **Kết quả bứt phá:** Accuracy nhảy vọt từ **48% lên 62.40%** (+14%) ngay cả khi chưa cần tăng thêm bất kỳ ảnh nào!
* **🗣️ Lời thoại thuyết trình mẫu:**
  > *"Stage 2 là bước ngoặt lớn nhất của đề tài. Nhóm nhận ra rằng vấn đề không phải mô hình không đủ mạnh, mà do nét chữ quá nhỏ bị nén co lại thành vệt mờ khi resize về 32x32. Nhờ thuật toán căn giữa Bounding Box và tẩy trắng nền scan, nét chữ giờ đây to rõ chiếm 80% canvas. Kết quả là độ chính xác tăng vọt +14% ngay lập tức mà chưa cần sinh thêm dữ liệu."*

---

### 📌 SLIDE 7: STAGE 3 - TĂNG CƯỜNG DỮ LIỆU & QUY TẮC MIỀN CẤM LẬT NGANG
* **Tiêu đề Slide:** Stage 3: Cân Bằng Dữ Liệu & Quy Tắc Miền Sống Còn
* **Nội dung hiển thị:**
  * Giải quyết bài toán mất cân bằng: Lớp `iv` (281 ảnh) gấp đôi lớp `ii` (157 ảnh), recall lớp `ii` ban đầu chỉ có 27%.
  * Data Augmentation chuyên biệt cho chữ viết tay: xoay nhẹ $\pm 8^\circ$, dịch chuyển $\pm 8\%$, zoom nhẹ $\pm 8\%$.
  * ⚠️ **Quy tắc miền sống còn (Domain Constraint):** **TUYỆT ĐỐI CẤM LẬT NGANG (`horizontal_flip=False`)**.
    * *Giải thích toán học:* Chữ số La Mã có tính bất đối xứng theo chiều ngang phản ánh giá trị toán học ($IV = 5 - 1 = 4$, nhưng khi lật ngang sẽ biến thành $VI = 5 + 1 = 6$). Lật ngang sẽ phá hỏng nhãn dữ liệu!
  * **Quy mô:** Cân bằng hoàn hảo 10 lớp, mỗi lớp 750 ảnh $\rightarrow$ Tổng quy mô **7.500 ảnh**.
  * **Kết quả:** Accuracy tăng lên **70.10%**, Recall lớp `ii` phục hồi mạnh từ 27% lên trên 70%.
* **🗣️ Lời thoại thuyết trình mẫu:**
  > *"Khi tăng cường dữ liệu ở Stage 3, nhóm đặt ra một nguyên tắc sống còn: Tuyệt đối không lật ngang ảnh. Vì trong chữ số La Mã, IV lật ngang sẽ thành VI, lật ngang sẽ làm đảo lộn nhãn toán học và biến dữ liệu thành rác. Nhờ tăng cường có kiểm soát và cân bằng 750 ảnh/lớp, mô hình đã chạm mốc 70.10%."*

---

### 📌 SLIDE 8: STAGE 4 - TỐI ƯU HÓA SÁT TRẦN (MAX CAPACITY)
* **Tiêu đề Slide:** Stage 4: Mở Rộng 9.500 Ảnh & Huấn Luyện Chuẩn 15 Epochs
* **Nội dung hiển thị:**
  * Khai thác tối đa hạn ngạch cuộc thi: Nâng quy mô lên **9.500 ảnh** (950 ảnh/lớp, giữ khoảng đệm an toàn 500 ảnh dưới mức trần 10.000 của ban tổ chức).
  * Chuẩn hóa đồng bộ 100% ảnh theo quy trình Stage 1 + Stage 2.
  * Giữ nguyên chuẩn **15 epochs** đồng bộ cho cả 5 Stages để làm thước đo đối chứng Ablation Study công bằng nhất.
  * **Kết quả đỉnh cao:**
    * Validation Accuracy đạt: **82.50%**
    * Label Book Accuracy đạt: **80.00%**
    * Lưu file trọng số tốt nhất: `best_vgg16_stage4.weights.h5`.
* **🗣️ Lời thoại thuyết trình mẫu:**
  > *"Ở Stage 4, nhóm khai thác tối đa trần dữ liệu cho phép với 9.500 ảnh chất lượng cao và huấn luyện kèm cơ chế EarlyStopping. Kết quả là mô hình VGG16 scratch đã hội tụ tối ưu và chạm ngưỡng 82.50% trên tập validation và 80% trên tập kiểm chuẩn độc lập."*

---

### 📌 SLIDE 9: STAGE 5 - BẢNG ĐỐI CHỨNG TOÀN DIỆN (ABLATION STUDY)
* **Tiêu đề Slide:** Stage 5: Tổng Kết Thực Nghiệm Đối Chứng & Ma Trận Nhầm Lẫn
* **Hình ảnh chèn:** [**`slide_ablation_study_chart.png`**](file:///e:/Group_DAT/deeplearningai-data-centric-competition/slide_ablation_study_chart.png)
* **Bảng tổng hợp tiến trình qua 5 Stage:**

| Giai đoạn | Can thiệp cốt tử trên Dữ liệu | Validation Acc | Label Book Acc | Mức tăng trưởng |
| :--- | :--- | :---: | :---: | :---: |
| **Stage 0** | Dữ liệu thô 2.880 ảnh (dính rò rỉ, ảnh rác, nét nhỏ) | 45.51% (ảo) | 36.54% | Mốc chuẩn ban đầu |
| **Stage 1** | Khử 49 ảnh rò rỉ MD5, xóa ảnh rác & lọc sai nhãn | 48.20% | 40.00% | Làm sạch thước đo |
| **Stage 2** | Căn giữa BBox 80% canvas + Tẩy trắng nền scan | **62.40%** | **58.00%** | **+14.2% (Đột phá lớn nhất)** |
| **Stage 3** | Cân bằng 10 lớp lên 7.500 ảnh (Cấm lật ngang) | **70.10%** | **68.00%** | +7.7% (Khử bias lớp) |
| **Stage 4** | Mở rộng 9.500 ảnh sát trần + Huấn luyện hội tụ sâu | **82.50%** | **80.00%** | **+12.4% (Chạm đỉnh)** |

* **🗣️ Lời thoại thuyết trình mẫu:**
  > *"Nhìn vào biểu đồ đối chứng trên slide, chúng ta thấy một đường cong tăng trưởng vững chắc từ 45.5% lên 82.5%. Trong đó, bước nhảy vọt lớn nhất thuộc về Stage 2 khi xử lý hình thái học của ảnh. Điều này chứng minh một cách định lượng sức mạnh của phương pháp tiếp cận Data-Centric AI."*

---

### 📌 SLIDE 10: TỔNG KẾT BÀI HỌC KINH NGHIỆM & KẾT LUẬN
* **Tiêu đề Slide:** Đúc Kết Bài Học Kinh Nghiệm Cho Môn Học DAT301m
* **3 bài học đắt giá:**
  1. *Hình thái học (Scale & Centering) quan trọng hơn số lượng:* Việc chữ chiếm bao nhiêu phần trăm diện tích và tương phản ra sao ảnh hưởng trực tiếp đến khả năng phân giải của các bộ lọc CNN.
  2. *Label Noise là kẻ thù số 1:* Làm sạch nhãn sai mang lại giá trị cao hơn nhiều so với việc cố nhồi nhét thêm dữ liệu bẩn.
  3. *Domain Knowledge quyết định Data Augmentation:* Hiểu bản chất bài toán giúp đặt ra các ràng buộc sống còn (như cấm lật ngang).
* **Thông điệp kết thúc:** *"Thay vì sửa mô hình, hãy sửa dữ liệu — đó là chìa khóa của AI hiện đại."*

---

## 🎯 PHẦN 2: BỘ CÂU HỎI VẤN ĐÁP PHẢN BIỆN (DEFENSE Q&A PLAYBOOK)

Dưới đây là 6 câu hỏi Giảng viên chắc chắn sẽ hỏi xoáy trong buổi thuyết trình và câu trả lời chuẩn xác nhất:

#### ❓ Câu hỏi 1: *"Tại sao nhóm tự build VGG16 mà không dùng thư viện Keras Applications có sẵn?"*
* **Trả lời phản biện:**  
  > *"Dạ thưa thầy, theo đúng yêu cầu của thầy và tinh thần môn học DAT301m, nhóm đã tự tay viết code từng layer cơ bản (13 lớp Conv2D chia làm 5 block và 3 lớp Dense phân loại) đúng chuẩn bài báo gốc của VGG16. Toàn bộ trọng số được khởi tạo ngẫu nhiên theo phân phối He-Normal, không sử dụng weights ImageNet có sẵn. Việc này nhằm đảm bảo mọi sự tăng trưởng độ chính xác đều phản ánh trung thực giá trị của dữ liệu chứ không dựa dẫm vào tri thức học sẵn từ ImageNet."*

#### ❓ Câu hỏi 2: *"Tại sao ở Stage 3 nhóm lại cấm lật ngang (horizontal flip) khi làm Data Augmentation?"*
* **Trả lời phản biện:**  
  > *"Dạ thưa thầy, chữ số La Mã có tính bất đối xứng theo chiều ngang phản ánh giá trị toán học: chữ IV là $5 - 1 = 4$, nhưng khi lật ngang sẽ biến thành chữ VI tức là $5 + 1 = 6$. Tương tự, IX ($10 - 1 = 9$) lật ngang sẽ biến thành XI ($10 + 1 = 11$). Nếu áp dụng horizontal flip, chúng ta sẽ vô tình làm đảo ngược nhãn dữ liệu (Data Corruption), khiến mô hình bị nhầm lẫn đối xứng trầm trọng. Đây là quy tắc miền bắt buộc phải tuân thủ."*

#### ❓ Câu hỏi 3: *"Yếu tố can thiệp dữ liệu nào mang lại bước nhảy vọt độ chính xác lớn nhất cho nhóm?"*
* **Trả lời phản biện:**  
  > *"Dạ thưa thầy, đó là kỹ thuật Bounding-Box Auto-Centering và Tẩy trắng nền ở Stage 2, mang lại mức tăng trưởng đột phá +14.2% accuracy. Trong dữ liệu gốc, chữ viết tay rất nhỏ nằm lệch ở góc, chỉ chiếm 15% diện tích. Khi nén về kích thước 32x32 của bài toán, nét chữ bị co lại thành vệt mờ 2 pixel khiến VGG16 không thể trích xuất đặc trưng. Việc cắt sát hộp bao và căn giữa giúp nét chữ chiếm 80% canvas vuông, to rõ ràng giúp các bộ lọc CNN nhận diện sắc nét từng gạch La Mã."*

#### ❓ Câu hỏi 4: *"Nhóm đã phát hiện ra các ảnh gán sai nhãn (Label Noise) bằng cách nào?"*
* **Trả lời phản biện:**  
  > *"Dạ thưa thầy, nhóm kết hợp hai phương pháp: Đầu tiên, nhóm viết script chạy Out-of-Fold Cross-Validation và trích xuất đặc trưng hình thái (độ tương phản, mật độ pixel tối và số nét liên thông). Thuật toán đã gắn cờ (flag) các mẫu có xác suất dự đoán mâu thuẫn cực mạnh với nhãn gán sẵn. Sau đó, nhóm trực tiếp mở file để kiểm tra mắt thường (Manual Inspection) và phát hiện chính xác các ảnh dị tật như ảnh trắng tinh không có nét bút nhưng dán nhãn VI, hoặc ảnh dấu X rất rõ ràng nhưng bị dán nhãn III."*

#### ❓ Câu hỏi 5: *"Tại sao nhóm lại dừng ở mức 9.500 ảnh ở Stage 4 mà không đẩy lên đúng 10.000 ảnh sát nút?"*
* **Trả lời phản biện:**  
  > *"Dạ thưa thầy, quy định đề bài là tổng số ảnh Train + Val không được vượt quá 10.000 ảnh. Nhóm chọn con số 9.500 ảnh (đúng 950 ảnh/lớp cho 10 lớp) để đảm bảo tính cân bằng hoàn hảo giữa các lớp, đồng thời giữ một khoảng đệm an toàn 500 ảnh để tuyệt đối không vi phạm luật thi đấu do sai số làm tròn batch size."*

#### ❓ Câu hỏi 6: *"Ý nghĩa của việc dùng EarlyStopping và ModelCheckpoint ở các Stage sau là gì?"*
* **Trả lời phản biện:**  
  > *"Dạ thưa thầy, theo đúng mã nguồn mẫu của ban tổ chức DeepLearning.AI, mạng nơ-ron sâu khi huấn luyện scratch sẽ có hiện tượng dao động ở các epoch cuối. Callbacks ModelCheckpoint giúp chúng em tự động lưu lại phiên bản trọng số đạt kết quả validation cao nhất trong toàn bộ quá trình huấn luyện, còn EarlyStopping giúp dừng sớm sau 8 epochs không cải thiện để tránh hiện tượng quá khớp và lãng phí tài nguyên tính toán."*

---
*Cẩm nang này được lập ra để đồng hành cùng slide Canva của nhóm 1, chúc nhóm đạt điểm 10 trọn vẹn trong buổi thuyết trình!*

