import torch
import torch.nn as nn
import torch.nn.functional as F


class FastViTBackbone(nn.Module):
    def __init__(self):
        super(FastViTBackbone, self).__init__()
        # Assume stem, stages, and final_conv represent the convolutional blocks described.
        self.stem = nn.Sequential(
            nn.Conv2d(24, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Additional layers according to the described structure...
        )
        # Simulating stages with depth-wise convolutions and linear layers as described.
        self.stages = nn.Sequential(
            # Blocks of Conv2d, ReLU, etc.
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            # Squeeze and Excitation blocks if needed
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, 2048)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.final_conv(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class TemporalHydra(nn.Module):
    def __init__(self):
        super(TemporalHydra, self).__init__()
        # Define layers for processing temporal information
        self.temporal_layers = nn.Sequential(
            # Linear layers and ReLU activations as placeholders
            nn.Linear(512, 1024),
            nn.ReLU(),
            # More layers as required...
        )
        # Output layers for each temporal output described
        self.plan = nn.Linear(1024, 4955)
        self.lead = nn.Linear(1024, 102)
        # More outputs as described...

    def forward(self, x):
        x = self.temporal_layers(x)
        outputs = {
            "plan": self.plan(x),
            "lead": self.lead(x),
            # Include other outputs...
        }
        return outputs


class Hydra(nn.Module):
    def __init__(self):
        super(Hydra, self).__init__()
        # Similar to TemporalHydra but for non-temporal processing
        self.layers = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            # Additional layers...
        )
        # Define output layers for each Hydra output
        self.meta = nn.Linear(1024, 48)
        self.desire_pred = nn.Linear(1024, 32)
        # More outputs as required...

    def forward(self, x):
        x = self.layers(x)
        outputs = {
            "meta": self.meta(x),
            "desire_pred": self.desire_pred(x),
            # Include other outputs...
        }
        return outputs


class ConvertModel(nn.Module):
    def __init__(self):
        super(ConvertModel, self).__init__()
        self.backbone = FastViTBackbone()
        self.temporal_hydra = TemporalHydra()
        self.hydra = Hydra()

    def forward(self, inputs):
        backbone_features = self.backbone(inputs["input_imgs"])
        temporal_outputs = self.temporal_hydra(backbone_features)
        hydra_outputs = self.hydra(backbone_features)
        # Merge outputs from both hydras
        outputs = {**temporal_outputs, **hydra_outputs}
        return outputs
