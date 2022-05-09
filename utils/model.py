import torch
import torch.nn as nn
import torchvision
from utils.cnn import CNN
from utils.utils import init_weights
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logger import step_flagging
from utils.wideResNet import WideResNet_encoder
from utils.preact_resnet import PreActResNet18
from utils.utils import count_parameters_in_MB
from utils import resnet


class MLPHead(nn.Module):
    def __init__(self, in_channels, mlp_hidden, projection_size, init_method='He'):
        super().__init__()

        mlp_hidden_size = round(mlp_hidden * in_channels)
        self.mlp_head = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size)
        )
        init_weights(self.mlp_head, init_method)

    def forward(self, x):
        return self.mlp_head(x)



class Encoder(nn.Module):
    def __init__(self, arch='cnn', num_classes=200, pretrained=True):
        super().__init__()
        if arch.lower().startswith('resnet') and arch in resnet.__all__:
            Resnet = resnet.__dict__[arch](pretrained=pretrained, in_channels=3)
            self.encoder = nn.Sequential(*list(Resnet.children())[:-1])
            self.feature_dim = Resnet.fc.in_features
        elif arch.startswith('cnn'):
            cnn = CNN(input_channel=3, n_outputs=num_classes)
            self.encoder = nn.Sequential(*list(cnn.children())[:-1])
            self.feature_dim = cnn.classifier.in_features
        elif arch == 'wideresnet':
            self.encoder = WideResNet_encoder(num_classes=num_classes)
            self.feature_dim = self.encoder.out_feature
        elif arch == 'preactresnet':
            preactResnet = PreActResNet18(num_classes = num_classes)
            self.encoder = nn.Sequential(*list(preactResnet.children())[:-1])
            self.feature_dim = preactResnet.linear.in_features
        else:
            raise AssertionError(f'{arch} is not supported!')
        count_parameters_in_MB(self.encoder)
    def forward(self, x):
        h = self.encoder(x)
        return h.view(h.shape[0], -1)


class Model(nn.Module):
    def __init__(self, arch='resnet18', num_classes=200, mlp_hidden=2, pretrained=True):
        super().__init__()
        self.encoder = Encoder(arch, num_classes, pretrained)
        self.classifier = MLPHead(self.encoder.feature_dim, mlp_hidden, num_classes)
        out_open = 2 * num_classes
        self.fc_open = nn.Linear(self.encoder.feature_dim, out_open, bias=False)

    def forward(self, x,has_open=True):
        x = self.encoder(x)
        logits = self.classifier(x)
        if has_open:
            logits_open = self.fc_open(x)
            return logits, logits_open
        else:
            return logits


if __name__=='__main__':
    for k,v in torchvision.models.__dict__.items():
        print(k)