import torch
from torch import nn


class FingerLoss(nn.Module):
    def __init__(self):
        super(FingerLoss, self).__init__()
        self.loss_fc = nn.CrossEntropyLoss()

    def forward(self, gesture_gt, landmark_gt, gesture, landmarks):

        class_loss = self.loss_fc(gesture, gesture_gt)
        l2_distant = torch.sum((landmark_gt - landmarks) * (landmark_gt - landmarks), axis=1)

        return torch.mean(class_loss * l2_distant), torch.mean(l2_distant)
