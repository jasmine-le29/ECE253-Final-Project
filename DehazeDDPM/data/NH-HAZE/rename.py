import os
import shutil

ROOT = r"./"

folders = ["hazy_train", "hazy_test", "gt_train", "gt_test"]

def clean_name(name):
    """Remove '_hazy' or '_gt' from filename, keep only prefix and .png"""
    if name.endswith("_hazy.png"):
        return name.replace("_hazy.png", ".png")
    if name.endswith("_GT.png"):
        return name.replace("_GT.png", ".png")
    return name

for folder in folders:
    folder_path = os.path.join(ROOT, folder)
    print(f"\nProcessing: {folder_path}")

    for fname in os.listdir(folder_path):
        old_path = os.path.join(folder_path, fname)
        new_name = clean_name(fname)
        new_path = os.path.join(folder_path, new_name)

        if old_path != new_path:
            print(f"  {fname}  →  {new_name}")
            # 如果你担心覆盖，可以先检查
            if os.path.exists(new_path):
                print(f"⚠ Warning: {new_name} already exists, skipping!")
                continue
            os.rename(old_path, new_path)

print("\n✅ 重命名完成！")
