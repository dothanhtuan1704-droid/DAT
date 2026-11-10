"""
🏛️ DAT301m Data-Centric AI Competition - Submission Validator
Kiểm tra tính hợp lệ của bộ dữ liệu theo quy định chính thức của BTC (DeepLearning.AI)
"""

import os
import glob
from PIL import Image

MAX_ALLOWED_IMAGES = 10000
EXPECTED_CLASSES = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]

def validate_submission_folder(folder_path="enhanced_data_max"):
    print("=" * 65)
    print(f"🔍 ĐANG KIỂM TRA TẬP DỮ LIỆU: '{folder_path}'")
    print("=" * 65)
    
    if not os.path.exists(folder_path):
        print(f"❌ LỖI: Không tìm thấy thư mục '{folder_path}'!")
        return False
        
    all_files = glob.glob(f"{folder_path}/**/*.png", recursive=True)
    total_count = len(all_files)
    print(f"📊 Tổng số ảnh tìm thấy: {total_count} ảnh")
    
    # 1. Kiểm tra giới hạn dung lượng
    if total_count > MAX_ALLOWED_IMAGES:
        print(f"❌ VI PHẠM QUY TẮC: Vượt quá giới hạn {MAX_ALLOWED_IMAGES} ảnh (Hiện có: {total_count})!")
        return False
    else:
        print(f"✅ QUY TẮC DUNG LƯỢNG: ĐẠT ({total_count} <= {MAX_ALLOWED_IMAGES} ảnh)")

    # 2. Kiểm tra danh sách 10 lớp
    found_classes = sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))])
    if set(found_classes) != set(EXPECTED_CLASSES):
        print(f"❌ LỖI NHÃN: Thiếu hoặc thừa nhãn lớp! Hiện có: {found_classes}")
        return False
    else:
        print(f"✅ QUY TẮC NHÃN: ĐẠT ĐỦ 10 LỚP CHỮ SỐ LA MÃ ({', '.join(EXPECTED_CLASSES)})")

    # 3. Kiểm tra tính toàn vẹn của file ảnh
    corrupt_count = 0
    blank_count = 0
    for f in all_files:
        try:
            with Image.open(f) as img:
                img.verify()
        except Exception:
            corrupt_count += 1
            
    if corrupt_count > 0:
        print(f"❌ LỖI TOÀN VẸN: Phát hiện {corrupt_count} ảnh bị hỏng!")
        return False
    else:
        print("✅ QUY TẮC TOÀN VẸN: 100% ảnh hợp lệ, không có file bị lỗi.")

    # 4. Thống kê phân bố lớp
    print("\n📈 PHÂN BỐ DỮ LIỆU CHI TIẾT TỪNG LỚP:")
    for c in EXPECTED_CLASSES:
        c_count = len(glob.glob(f"{folder_path}/{c}/*.png"))
        print(f"  • Lớp {c:<5}: {c_count:>5} ảnh ({c_count/total_count*100:.1f}%)")

    print("\n" + "=" * 65)
    print("🏆 KẾT LUẬN: BỘ DỮ LIỆU HOÀN TOÀN HỢP LỆ VÀ SẴN SÀNG NỘP ĐỒ ÁN!")
    print("=" * 65)
    return True

if __name__ == "__main__":
    validate_submission_folder("enhanced_data_max")
