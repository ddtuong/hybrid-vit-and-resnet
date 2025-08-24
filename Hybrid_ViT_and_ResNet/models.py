from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from configurations import VIT_NAME2MODEL, RESNET_NAME2MODEL

# HYBIRD ViT and ResNet 
# Combining ViT and ResNet at the feature level

class HybridViTResNet(nn.Module):
    def __init__(self, vit_name, resnet_name, num_classes, pretrained=True, progress=True, hidden_size=1024, dropout=0.25):
        super(HybridViTResNet, self).__init__()
        
        assert vit_name in VIT_NAME2MODEL.keys() and resnet_name in RESNET_NAME2MODEL.keys(), f"{vit_name} or {resnet_name} are not available."
       
       # Load pretrained models
        self.vit = VIT_NAME2MODEL[vit_name](pretrained=pretrained, progress=progress)
        self.res = RESNET_NAME2MODEL[resnet_name](pretrained=pretrained, progress=progress)
        
        # Remove the original fully connected layers to keep feature vectors only
        vit_features = self.vit.heads.head.in_features   # vit_features
        self.vit.heads = nn.Identity()
        
        res_features = self.res.fc.in_features           # res_features
        self.res.fc = nn.Identity()
        
        # New classifier on top of combined features
        in_features = vit_features + res_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        vit_out = self.vit(x)         # shape: (batch, vit_features)
        resnet_out = self.res(x)      # shape: (batch, res_features)

        # Concatenate feature vectors along the feature dimension
        combined = torch.cat((vit_out, resnet_out), dim=1)  # shape: (batch, vit_features + res_features)
        return self.classifier(combined)