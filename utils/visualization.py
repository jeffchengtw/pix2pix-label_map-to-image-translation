import torch
import cv2
import numpy as np
import os


def save_tensor_2(input_tensor, path, filename):
    # 檢查保存路徑是否存在，如果不存在則創建
    os.makedirs(path, exist_ok=True)
    
    for idx, each_image in enumerate(input_tensor):
        img_arr = tensor2im(each_image)
        file_path = os.path.join(path, f"{filename}_{idx}.jpg")
        cv2.imwrite(file_path, img_arr)

    

def tensor2im(image_tensor, imtype=np.uint8, normalize=False):

    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0      
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:        
        image_numpy = image_numpy[:,:,0]
    return image_numpy.astype(imtype)