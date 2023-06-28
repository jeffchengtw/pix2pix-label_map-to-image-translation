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



class MyDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.real, self.label = collect_file_paths(data_path)
        self.target_size = (128, 128) # w, h

        
    def __len__(self):
        return len(self.real)
    
    def __getitem__(self, index):
        real_path = self.real[index]
        label_path = self.label[index]

        # read from disk
        real_image = cv2.imread(real_path, cv2.IMREAD_UNCHANGED)
        label_image = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

        # image process
        processed_real = preprocess(real_image, self.target_size, 'resize')
        processed_label = preprocess(label_image, self.target_size, 'resize')
        #processed_label = preprocess_label_chapt(processed_label)
        # convert to tensor
        real_tensor = image_to_tensor(processed_real)
        label_tensor = image_to_tensor(processed_label)

        filename = os.path.splitext(os.path.basename(real_path))[0]

        return {
            'filename': filename,
            'real_tensor': real_tensor,
            'label_tensor': label_tensor
        }

        
