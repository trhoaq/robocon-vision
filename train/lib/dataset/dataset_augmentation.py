import os
import cv2
import numpy as np
import random
import torch
from albumentations.pytorch import ToTensorV2
import albumentations as A
from torch.utils.data import Dataset

# --- 1. Định nghĩa Pipeline Albumentations cho Object Detection ---

def get_train_augmentation_pipeline(patch_size=640):
    """
    Trả về một pipeline tăng cường dữ liệu mạnh mẽ của Albumentations cho object detection.

    Sử dụng:
        aug = get_train_augmentation_pipeline()
        out = aug(image=image_np, bboxes=bboxes_pascal_voc, class_labels=labels)
        image_t = out['image']
        bboxes_t = out['bboxes']
        labels_t = out['class_labels']
    """
    return A.Compose(
        [
            # --- Geometric (áp dụng đồng bộ cho image và bboxes) ---
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),

            # Giữ nguyên tỷ lệ khi resize
            A.LongestMaxSize(max_size=patch_size, p=1.0),
            
            # Pad để có kích thước vuông patch_size x patch_size
            A.PadIfNeeded(
                min_height=patch_size,
                min_width=patch_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=1.0
            ),

            # --- Photometric (chỉ ảnh) ---
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=0.5
                    ),
                    A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                    A.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5
                    ),
                    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
                ],
                p=1.0,
            ),

            # Noise / blur (chỉ ảnh)
            A.GaussNoise(p=0.2),
            A.GaussianBlur(p=0.3),
            
            # Chuẩn hóa về [0, 1] và chuyển sang PyTorch Tensor
            A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
            ToTensorV2(),
        ],
        # QUAN TRỌNG: Khai báo cách xử lý bounding box
        bbox_params=A.BboxParams(
            format='pascal_voc',                # Định dạng [x_min, y_min, x_max, y_max]
            label_fields=['class_labels'],      # Tên của list chứa class labels
            min_visibility=0.1                  # Loại bỏ bbox nếu dưới 10% diện tích còn lại
        )
    )

def get_val_augmentation_pipeline(patch_size=640):
    """
    Trả về pipeline TIỀN XỬ LÝ cho validation trong object detection.
    Không chứa các phép augmentation ngẫu nhiên.
    """
    return A.Compose(
        [
            # Giữ nguyên tỷ lệ khi resize
            A.LongestMaxSize(max_size=patch_size, p=1.0),
            
            # Pad để có kích thước vuông patch_size x patch_size
            A.PadIfNeeded(
                min_height=patch_size,
                min_width=patch_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=1.0
            ),
            
            # Chuẩn hóa về [0, 1] và chuyển sang PyTorch Tensor
            A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['class_labels']
        )
    )


# --- 2. Hàm Copy-Paste cho Object Detection ---

def copy_paste_detection(
    src_img: np.ndarray,
    src_bboxes: list,
    src_labels: list,
    dst_img: np.ndarray,
    dst_bboxes: list,
    dst_labels: list
):
    """
    Thực hiện "Copy-Paste" cho bài toán object detection.
    Copy một object ngẫu nhiên từ ảnh nguồn và dán vào ảnh đích.
    """
    if not src_bboxes:
        return dst_img, dst_bboxes, dst_labels

    # 1. Chọn một object ngẫu nhiên từ ảnh nguồn
    idx_to_paste = random.randint(0, len(src_bboxes) - 1)
    bbox_to_paste = src_bboxes[idx_to_paste]
    label_to_paste = src_labels[idx_to_paste]

    # Đảm bảo tọa độ là số nguyên
    x1, y1, x2, y2 = [int(c) for c in bbox_to_paste]

    # 2. Trích xuất patch của object
    obj_patch_img = src_img[y1:y2, x1:x2]
    patch_h, patch_w = obj_patch_img.shape[:2]
    
    # Bỏ qua nếu patch rỗng
    if patch_h == 0 or patch_w == 0:
        return dst_img, dst_bboxes, dst_labels

    # 3. Tìm vị trí ngẫu nhiên để dán trên ảnh đích
    dst_h, dst_w = dst_img.shape[:2]
    if patch_h > dst_h or patch_w > dst_w:
        return dst_img, dst_bboxes, dst_labels

    y_start = random.randint(0, dst_h - patch_h)
    x_start = random.randint(0, dst_w - patch_w)
    y_end = y_start + patch_h
    x_end = x_start + patch_w
    
    # 4. Dán patch vào ảnh đích
    dst_img[y_start:y_end, x_start:x_end] = obj_patch_img

    # 5. Thêm bounding box và label mới vào danh sách của ảnh đích
    new_bbox = [x_start, y_start, x_end, y_end]
    dst_bboxes.append(new_bbox)
    dst_labels.append(label_to_paste)

    return dst_img, dst_bboxes, dst_labels


# --- 3. Lớp Dataset cho Object Detection ---

class ObjectDetectionDataset(Dataset):
    """
    Dataset cho object detection, tải ảnh và annotation,
    áp dụng Copy-Paste và Albumentations.

    Giả định cấu trúc thư mục:
    - image_dir/
        - 001.png
        - 002.png
    - label_dir/
        - 001.txt
        - 002.txt
    
    Mỗi file .txt chứa các dòng: `class_id x_min y_min x_max y_max`
    """
    def __init__(self, image_dir, label_dir,
                 albumentations_pipeline=None, copy_paste_prob=0.5):

        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        # Giả định file label có tên tương ứng và đuôi .txt
        self.label_files = [os.path.splitext(f)[0] + '.txt' for f in self.image_files]

        self.transforms = albumentations_pipeline
        self.copy_paste_prob = copy_paste_prob

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        # 1. Tải ảnh và annotation ĐÍCH (ảnh chính)
        dst_img_path = os.path.join(self.image_dir, self.image_files[index])
        dst_label_path = os.path.join(self.label_dir, self.label_files[index])

        dst_image = cv2.imread(dst_img_path)
        dst_image = cv2.cvtColor(dst_image, cv2.COLOR_BGR2RGB)
        
        dst_bboxes, dst_labels = [], []
        if os.path.exists(dst_label_path):
            with open(dst_label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        dst_labels.append(int(parts[0]))
                        dst_bboxes.append([float(p) for p in parts[1:]])

        # 2. Quyết định có thực hiện Copy-Paste hay không
        if random.random() < self.copy_paste_prob:
            # 3. Tải ảnh và annotation NGUỒN (ảnh ngẫu nhiên)
            src_index = random.randint(0, len(self.image_files) - 1)
            src_img_path = os.path.join(self.image_dir, self.image_files[src_index])
            src_label_path = os.path.join(self.label_dir, self.label_files[src_index])
            
            src_image = cv2.imread(src_img_path)
            src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
            
            src_bboxes, src_labels = [], []
            if os.path.exists(src_label_path):
                 with open(src_label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            src_labels.append(int(parts[0]))
                            src_bboxes.append([float(p) for p in parts[1:]])
            
            # 4. Áp dụng Copy-Paste
            if src_bboxes:
                dst_image, dst_bboxes, dst_labels = copy_paste_detection(
                    src_image, src_bboxes, src_labels,
                    dst_image, dst_bboxes, dst_labels
                )

        # 5. Áp dụng pipeline albumentations
        final_image, final_bboxes, final_labels = dst_image, dst_bboxes, dst_labels
        if self.transforms:
            # Chuyển đổi list rỗng thành mảng numpy rỗng để pipeline hoạt động
            if not dst_bboxes:
                dst_bboxes = np.zeros((0, 4))
                
            augmented = self.transforms(image=dst_image, bboxes=dst_bboxes, class_labels=dst_labels)
            final_image = augmented['image']
            final_bboxes = augmented['bboxes']
            final_labels = augmented['class_labels']

        # 6. Định dạng output cho các model object detection của PyTorch
        target = {}
        # Đảm bảo bboxes là tensor ngay cả khi rỗng
        target["boxes"] = torch.as_tensor(final_bboxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(final_labels, dtype=torch.int64)

        return final_image, target
