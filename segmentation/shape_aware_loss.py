import torch
import torch.nn as nn


class ShapeAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, prediction, target):
        bce_loss = self.bce(prediction, target)

        prediction_prob = torch.sigmoid(prediction)

        intersection = (prediction_prob * target).sum(dim=(1, 2, 3))
        union = prediction_prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))

        dice = (2 * intersection + 1e-6) / (union + 1e-6)
        dice_loss = 1 - dice.mean()

        return bce_loss + dice_loss