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

class Dense_Layer(nn.Module):
    def __init__(self, in_channels, growthrate, bn_size):
        super(Dense_Layer, self).__init__()

        #self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, bn_size * growthrate, kernel_size=1, bias=False
        )

        #self.bn2 = nn.BatchNorm2d(bn_size * growthrate)
        self.conv2 = nn.Conv2d(
            bn_size * growthrate, growthrate, kernel_size=3, padding=1, bias=False
        )

    def forward(self, prev_features):
        out1 = torch.cat(prev_features, dim=1)
        out1 = self.conv1(F.relu(out1))
        out2 = self.conv2(F.relu(out1))
        return out2


class Dense_Block(nn.ModuleDict):
    def __init__(self, n_layers, in_channels, growthrate, bn_size):
        """
        A Dense block consists of `n_layers` of `Dense_Layer`
        Parameters
        ----------
            n_layers: Number of dense layers to be stacked
            in_channels: Number of input channels for first layer in the block
            growthrate: Growth rate (k) as mentioned in DenseNet paper
            bn_size: Multiplicative factor for # of bottleneck layers
        """
        super(Dense_Block, self).__init__()

        layers = dict()
        for i in range(n_layers):
            layer = Dense_Layer(in_channels + i * growthrate, growthrate, bn_size)
            layers['dense{}'.format(i)] = layer

        self.block = nn.ModuleDict(layers)

    def forward(self, features):
        if (isinstance(features, torch.Tensor)):
            #print(f'size is {features.size()}')
            features = [features]

        for _, layer in self.block.items():
            new_features = layer(features)
            features.append(new_features)

        return torch.cat(features, dim=1)


class UpGenerator(nn.Module):
    def __init__(self, in_channels=3,num_residuals=16):
        super().__init__()
        self.initial=nn.Sequential(
            nn.Conv2d(in_channels,64,kernel_size=3,stride=1,padding=1),
            nn.ReLU()
        )
        # self.resBlock=nn.Sequential(
        #     *[ResBlock(64) for _ in range(num_residuals)]
        # )
        self.denseblock=Dense_Block(10, 64, growthrate=32, bn_size=4)
        self.mid1=nn.Conv2d(in_channels=384,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.mid2 = nn.Conv2d(in_channels=128, out_channels=48, kernel_size=3, stride=1, padding=1)
        self.pixelSuffle=nn.PixelShuffle(2)
        # self.last=nn.Sequential(
        #     nn.PixelShuffle()
        #     #nn.ConvTranspose2d(in_channels=64,out_channels=256,kernel_size=3,padding=1,stride=2,output_padding=1),
        #     nn.ReLU(inplace=True),
        #     #nn.ConvTranspose2d(in_channels=256,out_channels=3,kernel_size=3,padding=1,stride=2,output_padding=1)
        # )
    def forward(self,x):
        x=self.initial(x)
        temp=x
        x=self.denseblock(x)
        x=temp+x
        x=self.mid1(x)
        x=self.mid2(x)
        x=self.pixelSuffle(x)
        x=self.pixelSuffle(x)
        return x



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
        self.inital=nn.Sequential(
            ConvBlock(in_channels,pool=False),
            ConvBlock(64,pool=True),
            ConvBlock(64,pool=True)
        )
        self.denseblock = Dense_Block(10, 64, growthrate=32, bn_size=4)
        self.mid1 = nn.Conv2d(in_channels=384, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.mid2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.last=nn.Conv2d(in_channels=64,out_channels=3,kernel_size=3,stride=1,padding=1)

    def forward(self,x):
        x=self.inital(x)
        temp=x
        x=self.denseblock(x)
        x=temp+x
        x=self.mid1(x)
        x=self.mid2(x)
        x=self.last(x)
        return x




def test():
    x=torch.randn((5,3,256,256))
    umodel=UpGenerator()
    dmodel=DownGenerator()
    output=umodel(x)
    print(output.shape)
    print(dmodel(output).shape)

if __name__=="__main__":
    test()
