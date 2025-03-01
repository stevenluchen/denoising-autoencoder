import torch
import torch.nn as nn
from torchvision.models import resnet18
import torchvision.datasets as dset
import torchvision.transforms as T

class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        
        base_model = resnet18(weights=None)
        base_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base_model.maxpool = nn.Identity()
        base_model.fc = nn.Identity()
        
        self.encoder = nn.Sequential(
            base_model,
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(256, 256 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(256, 4, 4)),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    