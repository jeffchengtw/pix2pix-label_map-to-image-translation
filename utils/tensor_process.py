import numpy as np
import torch


def image_to_tensor(input_arr: np.ndarray) -> torch.Tensor:
    if len(input_arr.shape) == 2:  # Single-channel image (H x W)
        input_arr = np.expand_dims(input_arr, axis=2)  # Add channel dimension
    
    # Normalize the array
    normalized_arr = input_arr.astype(np.float32) / 255.0

    # Convert the array to a tensor
    tensor = torch.from_numpy(np.transpose(normalized_arr, (2, 0, 1))).float()
    
    return tensor

def create_one_hot_label(label):
    size = label.size()
    one_hot_size = (size[0], 2, size[2], size[3])
    input_label = torch.cuda.FloatTensor(torch.Size(one_hot_size)).zero_()
    input_label = input_label.scatter_(1, label.data.long().cuda(), 1.0)
    input_label = input_label.detach().cpu()  # 將張量移回 CPU
    return input_label