"""
Author: byron
Descripttion: 
version: 
Date: 2022-07-27 10:05:28
LastEditors: way
LastEditTime: 2022-07-27 10:05:28
"""

import numpy as np
import cv2, os
from torchvision import datasets, transforms
from torch.utils import data


class FingerDatasets(data.Dataset):
    def __init__(self, file_list, path, transforms=None):
        self.line = None
        self.path = None

        self.transforms = transforms
        self.path = path
        with open(file_list, "r") as f:
            self.lines = f.readlines()

    def __getitem__(self, index):
        self.line = self.lines[index].strip().split()
        x1, y1, x2, y2 = np.asarray(self.line[2:6], dtype=np.int32)
        img_path = os.path.join(self.path, self.line[0])
        img = cv2.imread(img_path)[y1:y2, x1:x2, :]
        landmark = np.asarray(self.line[6:90], dtype=np.float32)
        if self.transforms:
            img = self.transforms(img)

        return (img, landmark, int(self.line[1]))

    def __len__(self):
        return len(self.lines)
