from functools import lru_cache
import torch
from torch.utils.data import Dataset
import tifffile
import numpy as np
import os
from roifile import ImagejRoi
import torchvision.transforms as T
import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import albumentations as A
from scipy.ndimage import distance_transform_edt
import random
import math
import shutil


class DNAClusterDataset(Dataset):
    def __init__(self, folder_paths, target_size=(500, 500), augment=True, max_dist=5, cache_images=True):
        self.folder_paths = folder_paths if isinstance(folder_paths, list) else [folder_paths]
        self.image_files = self._collect_images()
        self.target_size = target_size
        self.augment = augment
        self.max_dist = max_dist  # clamp distance

        if not self.image_files:
            raise ValueError("No TIFF images found in the provided folders.")

        self.resize_transform = T.Resize(self.target_size)

        # Augmentations
        self.augmentation = A.Compose([
            A.Rotate(limit=30, border_mode=0, crop_border=True, p=0.7),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            A.OneOf([
                A.Downscale(scale_range=(0.3, 0.9), p=1.0),
                A.AdvancedBlur(blur_limit=(3, 5), p=1.0)
            ], p=0.3),
            A.GaussNoise(std_range=(0.01, 0.05), mean_range=(0.0, 0.0), p=0.3),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
            A.RandomCrop(128, 128)
        ])

        self.errors_printed = {}

        if cache_images:
            print('Caching images. Disable if running out of RAM.')
            self.load_tiff_with_annotations = lru_cache()(self.load_tiff_with_annotations)

    def _collect_images(self):
        image_files = []
        for folder in self.folder_paths:
            if os.path.exists(folder):
                folder_images = [
                    os.path.join(folder, f) for f in os.listdir(folder)
                    if f.lower().endswith(".tif") and not f.startswith("._")
                ]
                image_files.extend(folder_images)
                print(f"Found {len(folder_images)} images in {folder}")
        print(f"Total images found: {len(image_files)}")
        return image_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        attempts = 0
        total_files = len(self.image_files)
        while attempts < total_files:
            tiff_path = self.image_files[idx]
            try:
                image, annotation_mask = self.load_tiff_with_annotations(tiff_path)
            except Exception as e:
                if str(e) not in self.errors_printed:
                    print(f"Error loading file {tiff_path}: {e}. Skipping.")
                self.errors_printed[str(e)] = True
                idx = (idx + 1) % total_files
                attempts += 1
                continue

            # If load_tiff_with_annotations returns None
            if image is None or annotation_mask is None:
                if tiff_path not in self.errors_printed:
                    print(f"Skipping file {tiff_path} because it returned None.")
                self.errors_printed[tiff_path] = True
                idx = (idx + 1) % total_files
                attempts += 1
                continue

            if image.shape[0] < 128 or image.shape[1] < 128:
                print(f"Skipping {tiff_path}: image too small ({image.shape}) for crop.")
                idx = (idx + 1) % total_files
                attempts += 1
                continue

            image_t = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0
            mask_t = torch.tensor(annotation_mask, dtype=torch.float32).unsqueeze(0)

            if self.augment:
                aug = self.augmentation(
                    image=image_t.numpy().squeeze(0),
                    mask=mask_t.numpy().squeeze(0)
                )
                aug_img = aug["image"]
                aug_mask = aug["mask"]

                image_t = torch.tensor(aug_img, dtype=torch.float32).unsqueeze(0)
                mask_t = torch.tensor(aug_mask, dtype=torch.float32).unsqueeze(0)

            mask_t = (mask_t > 0.5).float()

            mask_np = mask_t.squeeze(0).numpy()
            inverse_mask_np = (mask_np < 0.5).astype(bool)

            dist_map = distance_transform_edt(inverse_mask_np)
            if self.max_dist is not None and self.max_dist > 0:
                dist_map[dist_map > self.max_dist] = self.max_dist
                dist_map = 1 - dist_map / self.max_dist

            dist_map_t = torch.tensor(dist_map, dtype=torch.float32).unsqueeze(0)

            return image_t, mask_t, dist_map_t

        raise RuntimeError("No valid TIFF files found in dataset.")

    def load_tiff_with_annotations(self, tiff_path):
        with tifffile.TiffFile(tiff_path) as tif:
            image = tif.asarray()
            metadata = tif.imagej_metadata

        found_mask = False
        annotation_mask = np.zeros(image.shape, dtype=np.uint8)

        if metadata and 'Overlays' in metadata:
            roi_data = metadata['Overlays']
            if isinstance(roi_data, list):
                for roi_bytes in roi_data:
                    roi = ImagejRoi.frombytes(roi_bytes)
                    x, y = roi.coordinates().T
                    x = np.clip(x.astype(int), 0, image.shape[1]-1)
                    y = np.clip(y.astype(int), 0, image.shape[0]-1)
                    annotation_mask[y, x] = 1
                    found_mask = True

        if not found_mask:
            annotation_mask = None

        return image, annotation_mask


def split_tif_files(input_folders, output_folder_90, output_folder_10):
    # Create output folders if they don't exist
    os.makedirs(output_folder_90, exist_ok=True)
    os.makedirs(output_folder_10, exist_ok=True)

    for folder in input_folders:
        # Use the folder's basename to create subfolders in the outputs
        folder_name = os.path.basename(os.path.normpath(folder))
        subfolder_90 = os.path.join(output_folder_90, folder_name)
        subfolder_10 = os.path.join(output_folder_10, folder_name)
        os.makedirs(subfolder_90, exist_ok=True)
        os.makedirs(subfolder_10, exist_ok=True)

        # Find all tif files (case insensitive)
        tif_files = [f for f in os.listdir(folder) if f.lower().endswith('.tif')]
        if not tif_files:
            print(f"No TIFF files found in folder: {folder}")
            continue

        # Shuffle to randomize the selection
        random.shuffle(tif_files)

        n = len(tif_files)
        # Calculate 10% rounded up
        count_10 = math.ceil(n * 0.1)
        count_90 = n - count_10

        # Debug info (optional)
        print(f"Folder '{folder_name}': Total = {n}, 90% count = {count_90}, 10% count = {count_10}")

        # Copy files: first 90% to subfolder_90, remaining 10% to subfolder_10
        for file in tif_files[:count_90]:
            src = os.path.join(folder, file)
            dst = os.path.join(subfolder_90, file)
            shutil.copy2(src, dst)

        for file in tif_files[count_90:]:
            src = os.path.join(folder, file)
            dst = os.path.join(subfolder_10, file)
            shutil.copy2(src, dst)

