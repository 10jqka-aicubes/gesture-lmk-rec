import argparse
import time

import cv2, os
import numpy as np

import torch
from torchvision import transforms
from torch.utils import data
from torch.utils.data import DataLoader

from models import PFLDInference, CnnNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_img(files, pfld_backbone, cnnnet, val_dataloader):
    pfld_backbone.eval()
    cnnnet.eval()
    save_file = open(files, "w")
    with torch.no_grad():
        for img, name in val_dataloader:
            img = img.to(device)

            pfld_backbone = pfld_backbone.to(device)
            cnnnet = cnnnet.to(device)

            feature, landmarks = pfld_backbone(img)
            pred_y = cnnnet(feature)
            _, pred_y = torch.max(pred_y.cpu(), 1)
            pred_y = pred_y.numpy()[0]

            landmarks = landmarks.cpu().numpy()
            landmarks = landmarks.reshape(landmarks.shape[0], -1, 2)[0]  # landmark
            line = ""
            for x, y in landmarks:
                line = line + " " + str(x) + " " + str(y)
            save_file.write(name[0] + " " + str(pred_y) + " " + line + "\n")


def main(args):
    save_file = os.path.join(str(args.result_dir), "predict.txt")

    filename = os.path.join(str(args.model_dir), "checkpoint.pth.tar")
    checkpoint = torch.load(filename, map_location=device)
    pfld_backbone = PFLDInference().to(device)
    pfld_backbone.load_state_dict(checkpoint["pfld_backbone"])
    cnnnet = CnnNet().to(device)
    cnnnet.load_state_dict(checkpoint["cnnnet"])

    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([96, 96])])
    test_dataset = os.path.join(args.data_dir, "data/adata.txt")
    img_path = os.path.join(args.data_dir, "data/")
    fingerdataset = FingerDatasets(test_dataset, img_path, transform)
    dataloader = DataLoader(fingerdataset, batch_size=1, shuffle=False, num_workers=0)
    predict_img(save_file, pfld_backbone, cnnnet, dataloader)


def parse_args():
    parser = argparse.ArgumentParser(description="Predict")
    parser.add_argument("--model_dir", default="../model", type=str)
    parser.add_argument("--data_dir", default="../data/", type=str)
    parser.add_argument("--result_dir", default="../result/", type=str)

    args = parser.parse_args()
    return args


class FingerDatasets(data.Dataset):
    def __init__(self, file_list, path, transforms=None):
        self.line = None
        self.transforms = transforms
        self.path = path
        with open(file_list, "r") as f:
            self.lines = f.readlines()

    def __getitem__(self, index):
        self.line = self.lines[index].strip().split()

        x1, y1, x2, y2 = np.asarray(self.line[1:5], dtype=np.int32)

        img = cv2.imread(self.path + self.line[0])[y1:y2, x1:x2, :]
        if self.transforms:
            img = self.transforms(img)

        return (img, self.line[0])

    def __len__(self):
        return len(self.lines)


if __name__ == "__main__":
    args = parse_args()
    main(args)
