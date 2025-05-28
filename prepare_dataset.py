import os
import shutil
from sklearn.model_selection import train_test_split

def copy_files(file_list_path, src_root, dest_root):
    with open(file_list_path, 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            class_name = line.split('/')[0]
            src_path = os.path.join(src_root, line.replace('/', os.sep))
            dest_class_dir = os.path.join(dest_root, class_name)
            os.makedirs(dest_class_dir, exist_ok=True)
            try:
                shutil.copy(src_path, dest_class_dir)
            except FileNotFoundError:
                print(f"Dosya bulunamadı: {src_path}")

def split_train_val(train_dir, val_dir, split_ratio=0.2):
    for class_name in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_name)
        files = os.listdir(class_path)
        train_files, val_files = train_test_split(files, test_size=split_ratio, random_state=42)

        val_class_path = os.path.join(val_dir, class_name)
        os.makedirs(val_class_path, exist_ok=True)

        for file in val_files:
            src_file = os.path.join(class_path, file)
            shutil.move(src_file, os.path.join(val_class_path, file))

if __name__ == "__main__":
    src_images = "indoorCVPR_09/Images"
    os.makedirs("dataset/train", exist_ok=True)
    os.makedirs("dataset/test", exist_ok=True)
    os.makedirs("dataset/val", exist_ok=True)

    # Train ve test dosyalarını kopyala
    copy_files("TrainImages.txt", src_images, "dataset/train")
    copy_files("TestImages.txt", src_images, "dataset/test")

    # Train verisini %80 train, %20 val olarak böl
    split_train_val("dataset/train", "dataset/val")
    print("Veri başarıyla ayrıldı.")
