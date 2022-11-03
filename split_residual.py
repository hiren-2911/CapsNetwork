import torch
import torch.nn as nn
import torch.nn.functional as F



class basicResBlock(nn.Module):
    def __init__(self,channels=64):
        super().__init__()
        self.res=nn.Sequential(
            nn.Conv2d(channels,channels,kernel_size=1,stride=1,padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=1),
        )
    def forward(self,x):
        return x+self.res(x)

class ResBlock(nn.Module):
    def __init__(self,num_res=8,channels=64):
        super().__init__()
        self.block = nn.Sequential(basicResBlock(channels),
                                   basicResBlock(channels),
                                   basicResBlock(channels),
                                   basicResBlock(channels),
                                   basicResBlock(channels),
                                   basicResBlock(channels),
                                   basicResBlock(channels),
                                   basicResBlock(channels))

        # for i in range(num_res):
        #     self.block.append(basicResBlock(channels))

        self.last=nn.Conv2d(3*channels,channels,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        skip=x
        block1=self.block(x)
        block2=self.block(x)
        block3=self.block(x)
        concated=torch.concat([block1,block2,block3],1)
        last=self.last(concated)
        return skip+last


class GRL(nn.Module):
    def __init__(self,in_channels=1):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bicubic'),
            nn.Conv2d(in_channels,64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(16,1, kernel_size=1, stride=1, padding=0),
            # nn.ReLU(),
        )

    def forward(self, x):
        out=self.initial(x)
        return out

class Model(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.res1 = nn.Sequential(ResBlock(8,64))
        self.res2 = nn.Sequential(ResBlock(8, 128))
        self.res3 = nn.Sequential(ResBlock(8, 256))
        self.res4 = nn.Sequential(ResBlock(8, 512))
        # self.res5 = nn.Sequential(ResBlock(8, 1024))
        # self.res6 = nn.Sequential(ResBlock(8, 384))
        # self.res7 = nn.Sequential(ResBlock(8, 448))
        # self.res8 = nn.Sequential(ResBlock(8, 512))

        self.mid1=nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.mid2 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        self.grl=GRL()

        # deconvolution layers
        # self.deconv = nn.Sequential(
        #     nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=3 // 2, output_padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=3 // 2, output_padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 16, kernel_size=1),
        # )

        # reconstruction layer
        self.reconstruction = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        # self.mid2 = nn.Conv2d(in_channels=256, out_channels=48, kernel_size=3, stride=1, padding=1)
        #self.pixelSuffle = nn.PixelShuffle(2)
        self.up=nn.Sequential(
            #nn.PixelShuffle(),
            nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=3,padding=1,stride=2,output_padding=1),
            # nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=32,out_channels=16,kernel_size=3,padding=1,stride=2,output_padding=1)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)


    def forward(self, x):
        out = self.initial(x)
        res1 = self.res1(out)
        out1 = torch.cat([res1, out], 1)
        res2 = self.res2(out1)
        out2 = torch.cat([res2, out1], 1)
        res3 = self.res3(out2)
        out3 = torch.cat([res3, out2], 1)
        res4 = self.res4(out3)
        out4 = torch.cat([res4, out3], 1)
        # res5 = self.res5(out4)
        # out5 = torch.cat([res5, out4], 1)
        # res6 = self.res6(out5)
        # out6 = torch.cat([res6, out5], 1)
        # res7 = self.res7(out6)
        # out7 = torch.cat([res7, out6], 1)
        # res8 = self.res8(out7)
        # out8 = torch.cat([dcb8, out7], 1)
        out = self.mid1(out4)
        out = self.mid2(out)
        out = self.up(out)
        out = self.reconstruction(out)+self.grl(x)

        return out


def test():
    x=torch.randn((5,1,70,70))
    model=Model()
    output=model(x)
    print(output.shape)
    #print(dmodel(output).shape)

if __name__=="__main__":
    test()