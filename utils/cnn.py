from itertools import count
import torch.nn as nn
import numpy as np

def count_parameters_in_MB(model):
    ans = sum(np.prod(v.size()) for name, v in model.named_parameters())/1e6
    print('total parameters in the network are {} M'.format(ans))
    return ans

class CNN(nn.Module):
    def __init__(self, input_channel=3, n_outputs=10, dropout_rate=0.25, momentum=0.1):
        self.dropout_rate = dropout_rate
        self.momentum = momentum
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=self.momentum),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=self.momentum),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=self.momentum),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=self.momentum),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 196, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(196, momentum=self.momentum),
            nn.ReLU(),
            nn.Conv2d(196, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16, momentum=self.momentum),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Linear(256, n_outputs)
        
        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        return logits


if __name__ == '__main__':
    cnn = CNN()
    import hiddenlayer as hl
    import torch
    # hl_graph = hl.build_graph(cnn, torch.zeros([1, 3, 32, 32]))
    # hl_graph.theme = hl.graph.THEMES["blue"].copy()
    # hl_graph.save("pic.png", format="png")
    
    # from torchviz import make_dot
    # x = torch.randn(1, 3, 32, 32).requires_grad_(True)
    # viz = make_dot(cnn(x), params = dict(list(cnn.named_parameters()) + [('x', x)]))
    # viz.format = "png"
    # viz.directory = "output.png"
    # viz.view()
    
    encoder = nn.Sequential(*list(cnn.children())[:-1])
    print(count_parameters_in_MB(encoder))