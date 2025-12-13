import os
import numpy as np
from PIL import Image

def load_image_folder(folder_path):
    images = []
    labels = []

    # Label dictionary from folder names
    class_names = sorted(os.listdir(folder_path))
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}

    for cls_name in class_names:
        cls_folder = os.path.join(folder_path, cls_name)
        if not os.path.isdir(cls_folder):
            continue

        label = class_to_idx[cls_name]

        for img_name in os.listdir(cls_folder):
            img_path = os.path.join(cls_folder, img_name)

            img = Image.open(img_path).convert("RGB")
            img = np.array(img, dtype=np.float32)

            # Skip wrong shapes
            if img.shape != (32, 32, 3):
                print("Skipping:", img_path, img.shape)
                continue

            # Normalize
            img = img / 255.0

            images.append(img)
            labels.append(label)     # 🔥 مهم‌ترین بخش که تو نداشتی

    X = np.array(images)          # shape: (N, 32, 32, 3)
    y = np.array(labels)          # shape: (N,)

    return X, y, class_names


def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]


def accuracy(pred, true):
    return np.mean(pred == true)
