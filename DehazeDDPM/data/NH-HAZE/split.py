import os
import shutil
import random

# 根目录：放着所有 *_hazy.png 和 *_gt.png 的那个文件夹
ROOT = r"./NH-HAZE"

# 目标数量
NUM_TRAIN = 45  # 45 对做训练
# 其余自动做 val/test

# 1. 收集所有文件名
files = os.listdir(ROOT)
print(files)
hazy_files = [f for f in files if f.endswith("_hazy.png")]
gt_files   = [f for f in files if f.endswith("_GT.png")]

# 2. 根据前缀配对，确保每个前缀有 hazy 和 gt
def get_prefix(name):
    # 去掉后面的 "_hazy.png" 或 "_gt.png"
    if name.endswith("_hazy.png"):
        return name[:-len("_hazy.png")]
    elif name.endswith("_GT.png"):
        return name[:-len("_GT.png")]
    else:
        return None

hazy_dict = {get_prefix(f): f for f in hazy_files}
gt_dict   = {get_prefix(f): f for f in gt_files}

prefixes = sorted(set(hazy_dict.keys()) & set(gt_dict.keys()))

print(f"Found {len(prefixes)} pairs.")

if len(prefixes) < NUM_TRAIN:
    raise ValueError("训练数量设太大了，样本总数不够！")

# 3. 打乱并划分 train / val
random.shuffle(prefixes)
train_prefixes = prefixes[:NUM_TRAIN]
val_prefixes   = prefixes[NUM_TRAIN:]  # 这里应该是 10 对

print(f"Train pairs: {len(train_prefixes)}, Val/Test pairs: {len(val_prefixes)}")

# 4. 创建目标文件夹
hazy_train_dir = os.path.join(ROOT, "hazy_train")
gt_train_dir   = os.path.join(ROOT, "gt_train")
hazy_val_dir   = os.path.join(ROOT, "hazy_val")  # 你可以把 val 当 test 用
gt_val_dir     = os.path.join(ROOT, "gt_val")

for d in [hazy_train_dir, gt_train_dir, hazy_val_dir, gt_val_dir]:
    os.makedirs(d, exist_ok=True)

# 5. 拷贝（或移动）文件
def move_pair(prefixes, hazy_dst, gt_dst):
    for p in prefixes:
        hazy_src = os.path.join(ROOT, hazy_dict[p])
        gt_src   = os.path.join(ROOT, gt_dict[p])

        shutil.copy2(hazy_src, os.path.join(hazy_dst, hazy_dict[p]))
        shutil.copy2(gt_src,   os.path.join(gt_dst,   gt_dict[p]))

        # 如果你想“移动”而不是“复制”，改成 shutil.move 即可

move_pair(train_prefixes, hazy_train_dir, gt_train_dir)
move_pair(val_prefixes,   hazy_val_dir,   gt_val_dir)

print("Done! Files have been split into hazy/gt train & val.")
