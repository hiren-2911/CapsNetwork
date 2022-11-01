import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            #nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            #nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class GRL(nn.Module):
    def __init__(self,in_channels=1,scale=4):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Upsample(scale_factor=scale, mode='bicubic'),
            nn.Conv2d(in_channels,64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(16,1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )

    def forward(self, x):
        out=self.initial(x)
        return out

class UNET(nn.Module):
    def __init__(
            self, in_channels=1, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        #self.final_conv = nn.Conv2d(features[0], 3 , kernel_size=1)
        self.grl2=GRL(1,2)
        #self.grl4 = GRL(3, 4)
        self.last= nn.PixelShuffle(2)#nn.ConvTranspose2d(64, 63, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(17, 16, kernel_size=3,stride=1,padding=1)
        # self.final_layer2 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        self.reconstruction = nn.Conv2d(5, 1, kernel_size=3, padding=3 // 2)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)


    def forward(self, x):
        skip_connections = []
        temp=x
        for down in self.downs:
            x = down(x)

            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)



        x=self.last(x)   #16,140,140
        grl1=self.grl2(temp)
        concat_grl1 = torch.cat((grl1, x), dim=1) #17,140,140
        grl_last=self.grl2(grl1)
        #x = self.last(x)
        x=self.last(self.conv1(concat_grl1))
        concat_grl2 = torch.cat((grl_last, x), dim=1)
        #x=self.final_layer1(concat_grl2)
        return self.reconstruction(concat_grl2)

def test():
    x = torch.randn((3, 1, 70, 70))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)

if __name__ == "__main__":
    test()