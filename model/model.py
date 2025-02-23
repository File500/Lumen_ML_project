import torch.nn as nn


class MelanomaClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(MelanomaClassifier, self).__init__()
        self.model = nn.Sequential(

        )


    def forward(self, x):
        return self.model(x)