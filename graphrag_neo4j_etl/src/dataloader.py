import os

def process_folder(folder_path, current_demuc=None, current_vanban=None):
    folder_name = os.path.basename(folder_path)

    # Nếu thư mục là "demucX" (bắt đầu bằng "demuc"), cập nhật demuc
    if folder_name.startswith("demuc"):
        current_demuc = folder_name
        current_vanban = None  # Reset vanban khi chuyển sang demuc khác
        print(f"Entering demuc: {current_demuc}")

    # Nếu thư mục là "vanbanX" (bắt đầu bằng "vanban"), cập nhật vanban
    elif folder_name.startswith("vanban"):
        current_vanban = folder_name
        print(f"  Current vanban: {current_vanban}")

    # Duyệt qua các phần tử trong thư mục
    for entry in os.listdir(folder_path):
        entry_path = os.path.join(folder_path, entry)

        if os.path.isdir(entry_path):
            # Nếu là thư mục, tiếp tục đệ quy
            process_folder(entry_path, current_demuc, current_vanban)
        elif os.path.isfile(entry_path) and entry.endswith('.docx'):
            # Nếu là file, xử lý file
            print(f"    Processing file: {entry} in {current_demuc}/{current_vanban}")


# Hàm khởi chạy
def traverse_hierarchy(root_folder):
    print(f"Starting traversal from root: {root_folder}")
    process_folder(root_folder)

if __name__ == "__main__":
    # Thư mục gốc
    root_directory = "data"  # Đường dẫn đến thư mục cha
    traverse_hierarchy(root_directory)
