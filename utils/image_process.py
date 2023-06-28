import cv2
import numpy as np

def preprocess(input_arr: np.ndarray, output_size: tuple, mode: str = 'resize') -> np.ndarray:
    height, width = input_arr.shape[:2]
    target_height, target_width = output_size

    if mode == 'resize':
        # Resize the image to the target size
        output_arr = cv2.resize(input_arr, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    
    elif mode == 'padding':
        # Calculate the necessary padding
        pad_height = max(target_height - height, 0)
        pad_width = max(target_width - width, 0)
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        # Apply padding to the image
        output_arr = cv2.copyMakeBorder(input_arr, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)

    else:
        raise ValueError("Invalid mode. Supported modes are 'resize' and 'padding'.")

    return output_arr

