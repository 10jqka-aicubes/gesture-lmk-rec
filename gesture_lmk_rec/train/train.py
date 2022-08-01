#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse

import os

import numpy as np
import torch

from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from datasets import FingerDatasets
from models import PFLDInference, CnnNet
from loss import FingerLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)


def train(train_loader, pfld_backbone, cnnnet, criterion, optimizer, epoch):

    weighted_loss, loss = None, None
    for img, landmark_gt, y in train_loader:
        img = img.to(device)
        y = y.to(device)
        landmark_gt = landmark_gt.to(device)

        pfld_backbone = pfld_backbone.to(device)
        cnnnet = cnnnet.to(device)

        features, landmarks = pfld_backbone(img)
        pred = cnnnet(features)
        weighted_loss, loss = criterion(y, landmark_gt, pred, landmarks)
        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()
    return loss


def main(args):

    pfld_backbone = PFLDInference().to(device)
    cnnnet = CnnNet().to(device)
    criterion = FingerLoss()
    optimizer = torch.optim.Adam(
        [{"params": pfld_backbone.parameters()}, {"params": cnnnet.parameters()}],
        lr=args.base_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=args.lr_patience, verbose=True
    )

    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([96, 96])])
    file_list = os.path.join(args.train_data_dir, "data/train/train_data.txt")
    data_dir = os.path.join(args.train_data_dir, "data/train")
    fingerdataset = FingerDatasets(file_list, data_dir, transform)
    dataloader = DataLoader(fingerdataset, batch_size=args.train_batchsize, shuffle=False, num_workers=args.workers)
    max_loss = 10000
    for epoch in range(args.start_epoch, args.end_epoch + 1):
        train_loss = train(dataloader, pfld_backbone, cnnnet, criterion, optimizer, epoch)
        print("epoch:", epoch, "loss:", train_loss.detach().cpu().numpy())
        filename = os.path.join(str(args.model_dir), "checkpoint.pth.tar")
        if max_loss > train_loss:
            max_loss = train_loss
            save_checkpoint(
                {"epoch": epoch, "pfld_backbone": pfld_backbone.state_dict(), "cnnnet": cnnnet.state_dict()}, filename
            )


def parse_args():
    parser = argparse.ArgumentParser(description="pfld")
    # general
    parser.add_argument("-j", "--workers", default=0, type=int)

    ##  -- optimizer
    parser.add_argument("--base_lr", default=0.001, type=int)
    parser.add_argument("--weight-decay", "--wd", default=1e-6, type=float)

    # -- lr
    parser.add_argument("--lr_patience", default=25, type=int)

    # -- epoch
    parser.add_argument("--start_epoch", default=1, type=int)
    parser.add_argument("--end_epoch", default=25, type=int)

    parser.add_argument("--model_dir", default="../model", type=str)

    parser.add_argument("--train_data_dir", default="../data/train/", type=str)
    parser.add_argument("--train_batchsize", default=32, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
