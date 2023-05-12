import torch.nn as nn

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x.view(x.size(0), -1)

class ResidualPod(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(channels)
        self.ReLU = nn.ReLU(inplace = True)
        self.conv = nn.Conv2d(channels, channels, kernel_size = 3, stride = 1, padding = 1)
    
    def forward(self, x):
        x = self.bn(x)
        x = self.ReLU(x)
        x_pass = x
        x = self.conv(x)
        x = self.bn(x)
        x = self.ReLU(x)
        x = self.conv(x)
        return (x + x_pass)

class ResidualGroup(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ReLU = nn.ReLU(inplace = True)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv_in = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
        self.conv_trans = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1)
        self.conv_mid = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
        self.res_pod = ResidualPod(out_channels)

    def forward(self, x):
        x = self.bn1(x)
        x = self.ReLU(x)
        x_pass = self.conv_trans(x)
        x = self.conv_in(x)
        x = self.bn2(x)
        x = self.ReLU(x)
        x = self.conv_mid(x)
        x = x + x_pass
        x = self.res_pod(x)
        x = self.res_pod(x)
        return x

class Resnet(nn.Module):
    def __init__(self, n_input_channels, conv_channels : list[int]):
        super().__init__()

        self.model = nn.Sequential()
        self.model.append(nn.Conv2d(n_input_channels, conv_channels[0], kernel_size = 3, stride = 1, padding = 1))

        for i in range(len(conv_channels) - 1):
            self.model.append(ResidualGroup(conv_channels[i], conv_channels[i+1]))

        self.model.append(nn.Sequential(
            nn.BatchNorm2d(conv_channels[-1]),
            nn.ReLU(inplace = True),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
        ))

    def forward(self, x):
        return self.model(x)
