import os
import glob
import torch
import shutil

data_path = "/kaggle/input/rain-data/raw/rain1400/training"  # Change as needed
train_data_path = "/kaggle/working/rain1400_h5"

test_paths = {
    "Rain1400": {
        "rain": "/kaggle/input/rain-data/raw/rain1400/testing/rainy_image",
        "norain": "/kaggle/input/rain-data/raw/rain1400/testing/ground_truth"
    }
}

# Training config
batch_size = 18
epochs = 100
lr = 1e-3
recurrent_iter = 6
use_gpu = torch.cuda.is_available()
save_path = "/kaggle/working/PReNet_experiment"
save_freq = 1

src_dir = "/kaggle/input/checkpoint-prenet/output"
dst_dir = "/kaggle/working/PReNet_experiment"

os.makedirs(dst_dir, exist_ok=True)

files = glob.glob(os.path.join(src_dir, "*"))

for f in files:
    dst = os.path.join(dst_dir, os.path.basename(f))
    if not os.path.exists(dst):
        shutil.copy(f, dst)
        print(f"Copied {os.path.basename(f)}")
    else:
        print(f"{os.path.basename(f)} already exists")

use_curriculum = True
use_augmentation = True
cfg = {
    "E1": int(0.3 * epochs),
    "E2": int(0.7 * epochs),
}

os.makedirs(save_path, exist_ok=True)