import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self,channels=64):
        super().__init__()
        self.res=nn.Sequential(
            nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=1),
        )
    def forward(self,x):
        return x+self.res(x)



class DenseConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, growth_channels: int):
        super(DenseConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(int(growth_channels * 1), out_channels, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(int(growth_channels * 2), out_channels, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(int(growth_channels * 3), out_channels, (3, 3), (1, 1), (1, 1))
        self.conv5 = nn.Conv2d(int(growth_channels * 4), out_channels, (3, 3), (1, 1), (1, 1))
        self.conv6 = nn.Conv2d(int(growth_channels * 5), out_channels, (3, 3), (1, 1), (1, 1))
        self.conv7 = nn.Conv2d(int(growth_channels * 6), out_channels, (3, 3), (1, 1), (1, 1))
        self.conv8 = nn.Conv2d(int(growth_channels * 7), out_channels, (3, 3), (1, 1), (1, 1))

        self.relu = nn.ReLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.relu(self.conv1(x))

        out2 = self.relu(self.conv2(out1))
        out2_concat = torch.cat([out1, out2], 1)

        out3 = self.relu(self.conv3(out2_concat))
        out3_concat = torch.cat([out1, out2, out3], 1)

        out4 = self.relu(self.conv4(out3_concat))
        out4_concat = torch.cat([out1, out2, out3, out4], 1)

        out5 = self.relu(self.conv5(out4_concat))
        out5_concat = torch.cat([out1, out2, out3, out4, out5], 1)

        out6 = self.relu(self.conv6(out5_concat))
        out6_concat = torch.cat([out1, out2, out3, out4, out5, out6], 1)

        out7 = self.relu(self.conv7(out6_concat))
        out7_concat = torch.cat([out1, out2, out3, out4, out5, out6, out7], 1)

        out8 = self.relu(self.conv8(out7_concat))
        out8_concat = torch.cat([out1, out2, out3, out4, out5, out6, out7, out8], 1)

        return out8_concat

class UpGenerator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.initial=nn.Sequential(
            nn.Conv2d(in_channels,128,kernel_size=3,stride=1,padding=1),
            nn.ReLU()
        )
        self.dcb1 = nn.Sequential(DenseConvBlock(128, 16, 16))
        self.dcb2 = nn.Sequential(DenseConvBlock(256, 16, 16)) #256
        self.dcb3 = nn.Sequential(DenseConvBlock(384, 16, 16))
        self.dcb4 = nn.Sequential(DenseConvBlock(512, 16, 16))
        self.dcb5 = nn.Sequential(DenseConvBlock(640, 16, 16))
        self.dcb6 = nn.Sequential(DenseConvBlock(768, 16, 16))
        self.dcb7 = nn.Sequential(DenseConvBlock(896, 16, 16))
        self.dcb8 = nn.Sequential(DenseConvBlock(1024, 16, 16))
        self.bottleneck = nn.Sequential(
            nn.Conv2d(1152, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # deconvolution layers
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=3 // 2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=3 // 2, output_padding=1),
            nn.ReLU(inplace=True)
        )

        # reconstruction layer
        self.reconstruction = nn.Conv2d(256,3, kernel_size=3, padding=3 // 2)
       # self.mid2 = nn.Conv2d(in_channels=256, out_channels=48, kernel_size=3, stride=1, padding=1)
       # self.pixelSuffle = nn.PixelShuffle(2)
        # self.last=nn.Sequential(
        #     nn.PixelShuffle()
        #     #nn.ConvTranspose2d(in_channels=64,out_channels=256,kernel_size=3,padding=1,stride=2,output_padding=1),
        #     nn.ReLU(inplace=True),
        #     #nn.ConvTranspose2d(in_channels=256,out_channels=3,kernel_size=3,padding=1,stride=2,output_padding=1)
        # )
    def forward(self,x):
        out = self.initial(x)
        dcb1 = self.dcb1(out)
        out1 = torch.cat([dcb1, out], 1)
        dcb2 = self.dcb2(out1)
        out2 = torch.cat([dcb2, out1], 1)
        dcb3 = self.dcb3(out2)
        out3 = torch.cat([dcb3, out2], 1)
        dcb4 = self.dcb4(out3)
        out4 = torch.cat([dcb4, out3], 1)
        dcb5 = self.dcb5(out4)
        out5 = torch.cat([dcb5, out4], 1)
        dcb6 = self.dcb6(out5)
        out6 = torch.cat([dcb6, out5], 1)
        dcb7 = self.dcb7(out6)
        out7 = torch.cat([dcb7, out6], 1)
        dcb8 = self.dcb8(out7)
        out8 = torch.cat([dcb8, out7], 1)
        out=self.bottleneck(out8)
        out=self.deconv(out)
        out=self.reconstruction(out)


        return out



class ConvBlock(nn.Module):
    def __init__(self,in_channels,pool=False):
        super().__init__()
        self.conv=nn.Sequential(
            nn.MaxPool2d(kernel_size=2,stride=2)
            if pool
            else nn.Identity(),
            nn.Conv2d(in_channels,out_channels=64,kernel_size=3,padding=1,stride=1),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        return self.conv(x)

class DownGenerator(nn.Module):
    def __init__(self,in_channels=3):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.dcb1 = nn.Sequential(DenseConvBlock(128, 8, 8))
        self.dcb2 = nn.Sequential(DenseConvBlock(192, 8, 8))
        self.dcb3 = nn.Sequential(DenseConvBlock(256, 8, 8))
        self.dcb4 = nn.Sequential(DenseConvBlock(320, 8, 8))
        self.dcb5 = nn.Sequential(DenseConvBlock(384, 8, 8))
        self.dcb6 = nn.Sequential(DenseConvBlock(448, 8, 8))
        self.dcb7 = nn.Sequential(DenseConvBlock(512, 8, 8))
        self.dcb8 = nn.Sequential(DenseConvBlock(576, 8, 8))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(448, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            #nn.Conv2d(512, 256, kernel_size=1),
            #nn.ReLU(inplace=True)
        )


        self.conv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=3 // 2),
            nn.ReLU(inplace=True)
        )

        # reconstruction layer
        self.reconstruction = nn.Conv2d(64, 3, kernel_size=3, padding=3 // 2)

    def forward(self, x):
        out = self.initial(x)
        dcb1 = self.dcb1(out)
        out1 = torch.cat([dcb1, out], 1)
        dcb2 = self.dcb2(out1)
        out2 = torch.cat([dcb2, out1], 1)
        dcb3 = self.dcb3(out2)
        out3 = torch.cat([dcb3, out2], 1)
        dcb4 = self.dcb4(out3)
        out4 = torch.cat([dcb4, out3], 1)
        dcb5 = self.dcb5(out4)
        out5 = torch.cat([dcb5, out4], 1)
       #dcb6 = self.dcb6(out5)
       #out6 = torch.cat([dcb6, out5], 1)
       #dcb7 = self.dcb7(out6)
       #out7 = torch.cat([dcb7, out6], 1)
       #dcb8 = self.dcb8(out7)
       #out8 = torch.cat([dcb8, out7], 1)
        out = self.bottleneck(out5)
        out = self.conv(out)
        out = self.reconstruction(out)

        return out


class GRL(nn.Module):
    def __init__(self,in_channels=3):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bicubic'),
            nn.Conv2d(in_channels,64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(16,3, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )

    def forward(self, x):
        out=self.initial(x)
        return out





def test():
    x=torch.randn((5,3,256,256))
    umodel=UpGenerator()
    dmodel=DownGenerator()
    output=umodel(x)
    print(output.shape)
    print(dmodel(output).shape)

if __name__=="__main__":
    test()
