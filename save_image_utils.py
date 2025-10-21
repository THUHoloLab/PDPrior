import os
import numpy as np
from PIL import Image

def save_images(image, img_name, save_dir, max_index=np.inf):
    # image is a numpy array of shape NHWC
    # label is a numpy array of shape N
    for i in range(image.shape[0]):
        save_single_image(image[i], img_name[i], save_dir, max_index=max_index)

def save_single_image(image, img_name, save_dir, max_index=np.inf):
    # image is a numpy array of shape HWC
    # image = normalize_and_quantize(image)
    if image.ndim == 3 and image.shape[2] == 1:
        image = np.squeeze(image, axis=2)  # 去除第三个维度
    save_name = img_name
    Image.fromarray(image).save(os.path.join(save_dir, save_name))

