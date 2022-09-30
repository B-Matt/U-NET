from unet.architecture import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes

        self.in_conv = DoubleConv(n_channels, 64)
        self.down_conv_1 = DownConv(64, 128)
        self.down_conv_2 = DownConv(128, 256)
        self.down_conv_3 = DownConv(256, 512)
        self.down_conv_4 = DownConv(512, 1024)

        self.up_conv_1 = UpConv(1024, 512)
        self.up_conv_2 = UpConv(512, 256)
        self.up_conv_3 = UpConv(256, 128)
        self.up_conv_4 = UpConv(128, 64)
        self.out_conv = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down_conv_1(x1)
        x3 = self.down_conv_2(x2)
        x4 = self.down_conv_3(x3)
        x5 = self.down_conv_4(x4)

        x = self.up_conv_1(x5, x4)
        x = self.up_conv_2(x, x3)
        x = self.up_conv_3(x, x2)
        x = self.up_conv_4(x, x1)

        logits = self.out_conv(x)
        return logits #torch.sigmoid(logits)

def test():
    x = torch.randn((3, 1, 160, 160))
    model = UNet(1, 1, False)
    preds = model(x)

    print(preds.shape, x.shape)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()