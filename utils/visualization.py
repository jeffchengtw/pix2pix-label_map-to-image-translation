import torch
import math
from torchvision.utils import save_image
import cv2
import numpy as np
import os


def save_tensor_2(input_tensor, path, filename, epoch=None):
    # 檢查保存路徑是否存在，如果不存在則創建
    os.makedirs(path, exist_ok=True)
    
    for idx, (each_image, each_name) in enumerate(zip(input_tensor, filename)):
        img_arr = tensor2im(each_image)
        file_path = os.path.join(path, f"epoch_{epoch}_{each_name}_{idx}.jpg")
        cv2.imwrite(file_path, img_arr)

    

def tensor2im(image_tensor, imtype=np.uint8, normalize=True):

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

def create_grid_image_numpy(batch, grid_size):
    b, h, w = batch.shape[:3]
    grid_height = h * grid_size[0] + grid_size[0] - 1
    grid_width = w * grid_size[1] + grid_size[1] - 1
    grid_image = np.zeros((grid_height, grid_width), dtype=np.uint8)

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            if i * grid_size[1] + j < b:
                img = batch[i * grid_size[1] + j]
                start_x = (w + 1) * j
                start_y = (h + 1) * i
                end_x = start_x + w
                end_y = start_y + h
                grid_image[start_y:end_y, start_x:end_x] = img

                # 繪製垂直分隔線
                if j < grid_size[1] - 1:
                    grid_image[start_y:end_y, end_x] = 255

            # 繪製水平分隔線
            if i < grid_size[0] - 1:
                grid_image[end_y, :] = 255

    return grid_image

def create_grid_image(images, grid_size):
    num_images, height, width = images.shape

    rows = int(np.sqrt(num_images))
    cols = int(np.ceil(num_images / rows))

    grid = np.zeros((rows * height, cols * width))

    for i in range(num_images):
        row = i // cols
        col = i % cols
        grid[row * height: (row + 1) * height, col * width: (col + 1) * width] = images[i]

    return grid

def calculate_grid_size(n):
    sqrt_n = math.sqrt(n)
    rows = math.ceil(sqrt_n)
    cols = math.ceil(n / rows)
    return (rows, cols)

def save_feature_maps(batch_tensor, output_dir, filename, epoch):
    assert len(batch_tensor.shape) == 4, "Input tensor shape must be (batch_size, channels, height, width)"
    batch_size, num_channels, height, width = batch_tensor.shape

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 将张量转换为numpy数组并进行归一化
    tensor_np = batch_tensor.detach().cpu().numpy()
    tensor_np = (tensor_np - tensor_np.min()) / (tensor_np.max() - tensor_np.min())
    tensor_np *= 255

    # 逐个通道保存图像
    for idx, batch in enumerate(tensor_np):
        batch_numpy = batch.astype(np.uint8)
        b, h, w = batch_numpy.shape[:3]
        grid_size = calculate_grid_size(b)
        grid_image = create_grid_image_numpy(batch_numpy, grid_size=grid_size).astype(np.uint8)
        dst_dir = os.path.join(output_dir, f'{filename}_grid_image_{epoch}_{idx}.png')
        #save_image(grid_image, dst_dir)
        cv2.imwrite(dst_dir, grid_image)

def save_list_feature_maps(tensor_list, output_dir, filename, epoch):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    for idx, batch_tensor in enumerate(tensor_list):
        assert len(batch_tensor.shape) == 4, "Input tensor shape must be (batch_size, channels, height, width)"
        batch_size, num_channels, height, width = batch_tensor.shape

        # 将张量转换为numpy数组并进行归一化
        tensor_np = batch_tensor.detach().cpu().numpy()
        tensor_np = (tensor_np - tensor_np.min()) / (tensor_np.max() - tensor_np.min())
        tensor_np *= 255

        # 逐个通道保存图像
        for batch_idx, batch in enumerate(tensor_np):
            batch_numpy = batch.astype(np.uint8)
            grid_image = create_grid_image(batch_numpy, grid_size=8).astype(np.uint8)
            dst_dir = os.path.join(output_dir, f'{filename}_grid_image_{epoch}_{idx}_{batch_idx}.png')
            cv2.imwrite(dst_dir, grid_image)
