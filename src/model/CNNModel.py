import torch.nn as nn
from torchvision import models

class CNNModel:
    def __init__(self, num_labels):
        self.model = self.build_model(num_labels)

    def build_model(self, num_labels):
        model = models.resnet50(pretrained=True)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_labels),
            nn.Sigmoid()
        )
        return model
    
    def get_model(self):
        return self.model