import torch
from torch import nn
from torch.nn import functional as F
import torchvision

import sys
from math import sqrt


class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True)
                                  + 1e-8)


class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)


class EqualConvTranspose2d(nn.Module):
    ### additional module for OOGAN usage
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.ConvTranspose2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding, kernel_size2=None, padding2=None, pixel_norm=True):
        super().__init__()

        pad1 = padding
        pad2 = padding
        if padding2 is not None:
            pad2 = padding2

        kernel1 = kernel_size
        kernel2 = kernel_size
        if kernel_size2 is not None:
            kernel2 = kernel_size2

        convs = [EqualConv2d(in_channel, out_channel, kernel1, padding=pad1)]
        if pixel_norm:
            convs.append(PixelNorm())
        convs.append(nn.LeakyReLU(0.1))
        convs.append(EqualConv2d(out_channel, out_channel, kernel2, padding=pad2))
        if pixel_norm:
            convs.append(PixelNorm())
        convs.append(nn.LeakyReLU(0.1))

        self.conv = nn.Sequential(*convs)

    def forward(self, input):
        out = self.conv(input)
        return out


def upscale(feat):
    return F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=False)

class Generator(nn.Module):
    def __init__(self, input_code_dim=1024, in_channel=128, pixel_norm=True, tanh=True):  #128->1024
        super().__init__()
        self.input_dim = input_code_dim
        self.tanh = tanh
        self.input_layer = nn.Sequential(
            EqualConvTranspose2d(input_code_dim, in_channel, 4, 1, 0),
            PixelNorm(),
            nn.LeakyReLU(0.1))

        self.progression_4 = ConvBlock(in_channel, in_channel, 3, 1, pixel_norm=pixel_norm)
        self.progression_8 = ConvBlock(in_channel, in_channel, 3, 1, pixel_norm=pixel_norm)
        self.progression_16 = ConvBlock(in_channel, in_channel, 3, 1, pixel_norm=pixel_norm)
        self.progression_32 = ConvBlock(in_channel, in_channel, 3, 1, pixel_norm=pixel_norm)
        self.progression_64 = ConvBlock(in_channel, in_channel//2, 3, 1, pixel_norm=pixel_norm)
        self.progression_128 = ConvBlock(in_channel//2, in_channel//4, 3, 1, pixel_norm=pixel_norm)
        self.progression_256 = ConvBlock(in_channel//4, in_channel//4, 3, 1, pixel_norm=pixel_norm)

        self.to_rgb_8 = EqualConv2d(in_channel, 3, 1)
        self.to_rgb_16 = EqualConv2d(in_channel, 3, 1)
        self.to_rgb_32 = EqualConv2d(in_channel, 3, 1)
        self.to_rgb_64 = EqualConv2d(in_channel//2, 3, 1)
        self.to_rgb_128 = EqualConv2d(in_channel//4, 3, 1)
        self.to_rgb_256 = EqualConv2d(in_channel//4, 3, 1)
        
        self.max_step = 6

    def progress(self, feat, module):
        out = F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=False)
        out = module(out)
        return out

    def output(self, feat1, feat2, module1, module2, alpha):
        if 0 <= alpha < 1:
            skip_rgb = upscale(module1(feat1))
            out = (1-alpha)*skip_rgb + alpha*module2(feat2)
        else:
            out = module2(feat2)
        if self.tanh:
            return torch.tanh(out)
        return out

    def forward(self, input, step=0, alpha=-1):
        step=6
        if step > self.max_step:
            step = self.max_step
        temp = input.view(-1, self.input_dim, 1, 1)
        out_4 = self.input_layer(input.view(-1, self.input_dim, 1, 1))
        out_4 = self.progression_4(out_4)
        out_8 = self.progress(out_4, self.progression_8)
        if step==1:
            if self.tanh:
                return torch.tanh(self.to_rgb_8(out_8))
            return self.to_rgb_8(out_8)
        
        out_16 = self.progress(out_8, self.progression_16)
        if step==2:
            return self.output( out_8, out_16, self.to_rgb_8, self.to_rgb_16, alpha )
        
        out_32 = self.progress(out_16, self.progression_32)
        if step==3:
            return self.output( out_16, out_32, self.to_rgb_16, self.to_rgb_32, alpha )

        out_64 = self.progress(out_32, self.progression_64)
        if step==4:
            return self.output( out_32, out_64, self.to_rgb_32, self.to_rgb_64, alpha )
        
        out_128 = self.progress(out_64, self.progression_128)
        if step==5:
            return self.output( out_64, out_128, self.to_rgb_64, self.to_rgb_128, alpha )

        out_256 = self.progress(out_128, self.progression_256)
        if step==6:
            return self.output( out_128, out_256, self.to_rgb_128, self.to_rgb_256, alpha )


class Discriminator(nn.Module):
    def __init__(self, feat_dim=128):
        super().__init__()

        self.progression = nn.ModuleList([ConvBlock(feat_dim//4, feat_dim//4, 3, 1),
                                          ConvBlock(feat_dim//4, feat_dim//2, 3, 1),
                                          ConvBlock(feat_dim//2, feat_dim, 3, 1),
                                          ConvBlock(feat_dim, feat_dim, 3, 1),
                                          ConvBlock(feat_dim, feat_dim, 3, 1),
                                          ConvBlock(feat_dim, feat_dim, 3, 1),
                                          ConvBlock(feat_dim+1, feat_dim, 3, 1, 4, 0)])

        self.from_rgb = nn.ModuleList([EqualConv2d(3, feat_dim//4, 1),
                                       EqualConv2d(3, feat_dim//4, 1),
                                       EqualConv2d(3, feat_dim//2, 1),
                                       EqualConv2d(3, feat_dim, 1),
                                       EqualConv2d(3, feat_dim, 1),
                                       EqualConv2d(3, feat_dim, 1),
                                       EqualConv2d(3, feat_dim, 1)])

        self.n_layer = len(self.progression)

        self.linear = EqualLinear(feat_dim, 1)
        self.linear1024 = nn.Sequential(nn.Linear(feat_dim, 1024), nn.LeakyReLU(0.2))
    def forward(self, input, step=0, alpha=-1):
        step=3
        for i in range(step, -1, -1):
            index = self.n_layer - i - 1

            if i == step:
                out = self.from_rgb[index](input)

            if i == 0:
                out_std = torch.sqrt(out.var(0, unbiased=False) + 1e-8)
                mean_std = out_std.mean()
                mean_std = mean_std.expand(out.size(0), 1, 4, 4)
                out = torch.cat([out, mean_std], 1)

            out = self.progression[index](out)

            if i > 0:
                # out = F.avg_pool2d(out, 2)
                out = F.interpolate(out, scale_factor=0.5, mode='bilinear', align_corners=False)

                if i == step and 0 <= alpha < 1:
                    # skip_rgb = F.avg_pool2d(input, 2)
                    skip_rgb = F.interpolate(input, scale_factor=0.5, mode='bilinear', align_corners=False)
                    skip_rgb = self.from_rgb[index + 1](skip_rgb)
                    out = (1 - alpha) * skip_rgb + alpha * out

        out = out.squeeze(2).squeeze(2)
        # print(input.size(), out.size(), step)
        #out = self.linear(out)
        out_final = self.linear1024(out)
        return out_final

EPS = 1e-7
class PerceptualLoss(nn.Module):
    def __init__(self, requires_grad=False, loss_type='l2', n_scale=1, slice_indices=[2]):
        super(PerceptualLoss, self).__init__()
        mean_rgb = torch.FloatTensor([0.485, 0.456, 0.406])
        std_rgb = torch.FloatTensor([0.229, 0.224, 0.225])
        self.register_buffer('mean_rgb', mean_rgb)
        self.register_buffer('std_rgb', std_rgb)
        self.loss_type = loss_type
        self.n_scale = n_scale
        self.slice_indices = slice_indices[:]

        vgg_pretrained_features = torchvision.models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        self.classifier = nn.Conv2d(256, 1, 1, 1, bias=False)

    def normalize(self, x):
        out = x/2 + 0.5
        out = (out - self.mean_rgb.view(1,3,1,1)) / self.std_rgb.view(1,3,1,1)
        return out

    def compute_loss(self, f1, f2):
        if self.loss_type == 'l1':
            loss = (f1-f2).abs()
        elif self.loss_type == 'cosine':
            f1, f2 = F.normalize(f1,p=2,dim=1), F.normalize(f2,p=2,dim=1)
            loss = (f1-f2)**2
            loss = loss.sum(1, keepdim=True)
        else:
            loss = (f1-f2)**2
        return loss

    def resize(self, tensor, size):
        h, w = size
        h0, w0 = tensor.shape[2:]
        if h0==h and w0==w:
            return tensor
        assert h0%h==0 and w0%w==0
        sh, sw = h0//h, w0//w
        out = nn.functional.avg_pool2d(tensor, kernel_size=(sh,sw), stride=(sh,sw))
        return out

    def __call__(self, im1, im2, mask=None):
        im = torch.cat([im1,im2], 0)
        im = self.normalize(im)  # normalize input
        losses = []
        input = im
        for i in range(self.n_scale):
            if i == 0:
                input = input
            else:
                input = F.avg_pool2d(input, 2, stride=2, padding=0)
            ## compute features
            feats = []
            f = self.slice1(input)
            feats += [torch.chunk(f, 2, dim=0)]
            f = self.slice2(f)
            feats += [torch.chunk(f, 2, dim=0)]
            f = self.slice3(f)
            feats += [torch.chunk(f, 2, dim=0)]
            f = self.slice4(f)
            feats += [torch.chunk(f, 2, dim=0)]

            for idx in self.slice_indices:  # use relu3_3 features only
                f1,f2 = feats[idx]
                loss = self.compute_loss(f1,f2)
                if mask is not None:
                    b, c, h, w = loss.shape
                    _, _, hm, wm = mask.shape
                    sh, sw = hm//h, wm//w
                    mask0 = nn.functional.avg_pool2d(mask, kernel_size=(sh,sw), stride=(sh,sw)).expand_as(loss)
                    loss = (loss * mask0).sum() / mask0.sum()
                else:
                    loss = loss.mean()
                losses += [loss]
        return sum(losses) / len(losses)



def test():
    x=torch.randn((64,3,32,32))
    Decoder=Generator()
    Encoder=Discriminator()
    latent=Encoder(x)
    output=Decoder(latent)
    print(f"Latent space {latent.size()}")
    print(f"Output space {output.size()}")


if __name__=="__main__":
    test()