import torch
import torch.nn as nn


class DHKFHNet(nn.Module):

    def __init__(
        self,
        feature_size=13,
        num_classes=2
    ):
        super().__init__()

        self.image_features = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                3,
                padding=1
            ),

            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                3,
                padding=1
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                3,
                padding=1
            ),

            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                128,
                256,
                3,
                padding=1
            ),

            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                256 + feature_size,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.4),

            nn.Linear(
                128,
                num_classes
            )
        )

    def forward(
        self,
        image,
        features
    ):

        image_features = self.image_features(
            image
        )

        image_features = torch.flatten(
            image_features,
            start_dim=1
        )

        combined = torch.cat(
            [
                image_features,
                features
            ],
            dim=1
        )

        return self.classifier(
            combined
        )