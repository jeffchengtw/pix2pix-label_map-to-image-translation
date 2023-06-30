import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from utils.utils import*
from utils.tensor_process import*
from utils.image_process import*

def preprocess_label_chapt(input: np.ndarray) -> np.ndarray:
    
    if 255 in input:
        output = np.where(input == 255, 0, input + 1)
    classes = np.unique(output)
    n_classes = len(classes)
    output = np.searchsorted(classes, output).astype(np.uint8)
    return output

def check_labelnc(image_paths: 'list[str]') -> 'list':
    pixel_set = set()
    for path in image_paths:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        arr = np.array(image)
        unique_pixels = np.unique(arr)
        pixel_set.update(unique_pixels)
    return sorted(list(pixel_set))


class MyDataset(Dataset):
    def __init__(self, data_path, phase, target_size):
        self.phase = phase
        self.target_size = target_size  # w, h
        self.label_nc = 0

        if self.phase == 'train':
            self.real = collect_image_files(os.path.join(data_path, 'train_real'))
            self.label = collect_image_files(os.path.join(data_path, 'train_label'))
        elif self.phase == 'test':
            self.label = collect_image_files(os.path.join(data_path, 'test_label'))
        else:
            raise ValueError("Invalid phase. Supported values are 'train' and 'test'.")
        
        self.label_nc = len(check_labelnc(self.label))

        
    def __len__(self):
        if self.phase == 'train':
            return len(self.real)
        elif self.phase == 'test':
            return len(self.label)
    
    def __getitem__(self, index):
        if self.phase == 'train':
            real_path = self.real[index]
            label_path = self.label[index]
        elif self.phase == 'test':
            label_path = self.label[index]

        # read from disk
        real_image = cv2.imread(real_path, cv2.IMREAD_UNCHANGED) if self.phase == 'train' else None
        label_image = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

        # image process
        processed_real = preprocess(real_image, self.target_size, 'resize') if self.phase == 'train' else None
        processed_label = preprocess(label_image, self.target_size, 'resize')
        processed_label = preprocess_label_chapt(processed_label)
        
        # convert to tensor
        real_tensor = image_to_tensor(processed_real) if self.phase == 'train' else None
        label_tensor = create_one_hot_label(processed_label, self.label_nc)


        filename = os.path.splitext(os.path.basename(label_path))[0]

        if self.phase == 'train':
            return {
                'filename': filename,
                'real_tensor': real_tensor,
                'label_tensor': label_tensor
            }
        elif self.phase == 'test':
            return {
                'filename': filename,
                'label_tensor': label_tensor
            }

        
