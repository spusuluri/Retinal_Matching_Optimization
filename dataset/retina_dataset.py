import torch
import scipy.stats as st
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image
import numpy as np
import imgaug.augmenters as iaa
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob

from common.common_util import pre_processing
from common.train_util import get_gaussian_kernel
import time

class RetinaDataset(Dataset):
    def __init__(self, data_path, split_file='eccv22_train.txt',
                 is_train=True, data_shape=(768, 768), auxiliary=None):
        self.data = []

        self.image_path = os.path.join(data_path, 'Images')
        self.label_path = os.path.join(data_path, 'Ground Truth')
        # self.split_file = os.path.join(data_path, 'ImageSets', split_file)

        self.enhancement_sequential = iaa.Sequential([
            iaa.Multiply((1.0, 1.2)),  # change brightness, doesn't affect keypoints
            iaa.Sometimes(
                0.2,
                iaa.GaussianBlur(sigma=(0, 6))
            ),
            iaa.Sometimes(
                0.2,
                iaa.LinearContrast((0.75, 1.2))
            ),
        ], random_order=True)

        self.is_train = is_train
        self.model_image_height, self.model_image_width = data_shape[0], data_shape[1]
        
        for file in glob.glob("./data/FIRE/Images/*.jpg"):
            filename = os.path.basename(file)
            sample_id = filename.split("_")[0]
            sample_num = filename.split(".")[0].split("_")[1]
            label = f"control_points_{sample_id}_1_2.txt"
            label = os.path.join(self.label_path, label)
            self.data.append((file, label, sample_num))



        self.transforms = transforms.Compose([
            transforms.Resize((self.model_image_width, self.model_image_height)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_with_label = True
        image_path, label_path, num = self.data[index]

        label_name = '[None]'

        image = Image.open(image_path).convert('RGB')
        
        image = np.asarray(image)
        image = image[:, :, 1]

        image = pre_processing(image)
        # if self.is_train:
        #     image = self.enhancement_sequential(image=image)
        image = Image.fromarray(image)

        label_name = os.path.split(label_path)[-1]
        keypoint_position = np.loadtxt(label_path)  # (2, n): (x, y).T
        
        if num == 1:
            keypoint_position = keypoint_position[:, 0:2]
        elif num == 2:
            keypoint_position = keypoint_position[:, 2:]
        

        keypoint_position[:, 0] *= self.model_image_height / image.size[-1]
        keypoint_position[:, 1] *= self.model_image_width / image.size[-2]

        tensor_position = torch.zeros([self.model_image_height, self.model_image_width])

        tensor_position[keypoint_position[:, 1], keypoint_position[:, 0]] = 1
        tensor_position = tensor_position.unsqueeze(0)
        input_with_label = True
        image_tensor = self.transforms(image)

        return image_tensor, input_with_label, tensor_position, label_name



