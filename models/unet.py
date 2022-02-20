import torch
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):

    def __init__(self, in_channels, classes=2, activate=True):
        super(UNet, self).__init__()

        # filters = [32, 64, 128, 256, 512]
        filters = [64, 128, 256, 512, 1024]

        self.in_channels = in_channels
        self.classes = classes
        self.activate = activate

        self.input = DoubleConv(in_channels, filters[0])
        self.pool1 = nn.MaxPool2d(2)
        self.conv1 = DoubleConv(filters[0], filters[1])
        self.pool2 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(filters[1], filters[2])
        self.pool3 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(filters[2], filters[3])
        self.pool4 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(filters[3], filters[4])

        self.up1 = nn.ConvTranspose2d(filters[4], filters[3], 2, stride=2)
        self.conv5 = DoubleConv(filters[4], filters[3])
        self.up2 = nn.ConvTranspose2d(filters[3], filters[2], 2, stride=2)
        self.conv6 = DoubleConv(filters[3], filters[2])
        self.up3 = nn.ConvTranspose2d(filters[2], filters[1], 2, stride=2)
        self.conv7 = DoubleConv(filters[2], filters[1])
        self.up4 = nn.ConvTranspose2d(filters[1], filters[0], 2, stride=2)
        self.conv8 = DoubleConv(filters[1], filters[0])

        self.output = nn.Conv2d(filters[0], classes, kernel_size=1)

    def forward(self, x):
        input = self.input(x)  # 1->32
        p1 = self.pool1(input)
        c1 = self.conv1(p1)  # 32->64
        p2 = self.pool2(c1)
        c2 = self.conv2(p2)  # 64->128
        p3 = self.pool3(c2)
        c3 = self.conv3(p3)  # 128->256
        p4 = self.pool4(c3)
        c4 = self.conv4(p4)  # 256->512
        up1 = self.up1(c4)  # 512->256
        c5 = self.conv5(torch.cat([up1, c3], dim=1))
        up2 = self.up2(c5)
        c6 = self.conv6(torch.cat([up2, c2], dim=1))
        up3 = self.up3(c6)
        c7 = self.conv7(torch.cat([up3, c1], dim=1))
        up4 = self.up4(c7)
        c8 = self.conv8(torch.cat([up4, input], dim=1))
        out = self.output(c8)
        return nn.Softmax(dim=1)(out) if self.activate else out