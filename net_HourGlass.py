import torch
from torch.nn.utils import weight_norm as wn
import torch.nn as nn
from pytorch_wavelets import DWTForward, DWTInverse


def conv_layer(inc, outc, kernel_size=3, groups=1, bias=False, negative_slope=0.2, activation=False,
               weight_normalization=True):
    layers = []

    if activation == 'before':
        layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))

    m = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                  groups=groups, bias=bias, stride=1)

    if weight_normalization:
        layers.append(wn(m))
    else:
        layers.append(m)

    if activation == 'after':
        layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))

    return nn.Sequential(*layers)


class NOResBlock(nn.Module):
    def __init__(self, in_c):
        super(NOResBlock, self).__init__()
        self.conv1 = conv_layer(in_c, in_c, kernel_size=3, groups=1, bias=True, negative_slope=0.2,
                                activation=False, weight_normalization=True)

        self.conv2 = conv_layer(in_c, in_c, kernel_size=3, groups=1, bias=True, negative_slope=1,
                                activation=False, weight_normalization=True)

    def forward(self, x):
        return self.conv2(self.conv1(x)) + x


class Bottleneck(nn.Module):
    expansion = 2
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, bias=True)

        self.conv2 = nn.Conv2d(inplanes + planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)

        self.conv3 = nn.Conv2d(inplanes + planes + planes, planes * 2, kernel_size=3, padding=1, bias=True)

        self.relu = nn.PReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(self.relu(x))
        outc = torch.cat((x, out), dim=1)

        out = self.conv2(self.relu(outc))
        outc = torch.cat((outc, out), dim=1)

        out = self.conv3(self.relu(outc))

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

        self.DWT = DWTForward(J=1, wave='haar')
        self.IDWT = DWTInverse(wave='haar')

        self.conv256_64 = nn.Conv2d(256, 64, kernel_size=1)
        self.conv192_48 = nn.Conv2d(192, 48, kernel_size=1)
        self.conv48_12 = nn.Conv2d(48, 12, kernel_size=1)

        self.conv16_64 = nn.Conv2d(16, 64, kernel_size=1)
        self.conv12_48 = nn.Conv2d(12, 48, kernel_size=1)
        self.conv3_12 = nn.Conv2d(3, 12, kernel_size=1)

        self.convtran64 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=1)

    def _transformer(self, DMT1_yl, DMT1_yh):
        list_tensor = []
        for i in range(3):
            list_tensor.append(DMT1_yh[0][:, :, i, :, :])
        list_tensor.append(DMT1_yl)
        return torch.cat(list_tensor, 1)

    def _Itransformer(self, out):
        yh = []
        C = int(out.shape[1] / 4)
        y = out.reshape((out.shape[0], C, 4, out.shape[-2], out.shape[-1]))
        yl = y[:, :, 0].contiguous()
        yh.append(y[:, :, 1:].contiguous())
        return yl, yh

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes * block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n - 1][0](x)

        ll, _ = self.DWT(x)
        low1 = self._transformer(ll, _)
        if ll.size()[1] == 64:
            downc = self.conv256_64
        elif ll.size()[1] == 48:
            downc = self.conv192_48
        elif ll.size()[1] == 12:
            downc = self.conv48_12

        low1 = downc(low1)
        low1 = self.hg[n - 1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n - 1, low1)
        else:
            low2 = self.hg[n - 1][3](low1)

        low3 = self.hg[n - 1][2](low2)

        if low3.size()[1] == 64:
            upc = self.conv16_64
        elif low3.size()[1] == 48:
            upc = self.conv12_48
        elif low3.size()[1] == 12:
            upc = self.conv3_12

        up2 = upc(self.IDWT(self._Itransformer(low3)))

        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.DWT = DWTForward(J=1, wave='haar')
        self.IDWT = DWTInverse(wave='haar')

        hgm1, hgm2, hgm3 = [], [], []
        for i in range(3):
            hgm1.append(Hourglass(Bottleneck, num_blocks=1, planes=6, depth=4))
        for i in range(6):
            hgm2.append(Hourglass(Bottleneck, num_blocks=1, planes=24, depth=4))
        for i in range(9):
            hgm3.append(Hourglass(Bottleneck, num_blocks=1, planes=32, depth=4))
        self.HGM_1 = nn.ModuleList(hgm1)
        self.HGM_2 = nn.ModuleList(hgm2)
        self.HGM_3 = nn.ModuleList(hgm3)

        self.conv8x = nn.Sequential(
            conv_layer(192, 64),
            conv_layer(64, 64)
        )
        self.conv4x = NOResBlock(48)
        
        self.conv2x = conv_layer(12, 12)

        self.downc8 = nn.Sequential(
            conv_layer(576, 128),
            conv_layer(128, 64))

        self.conv_8_4 = nn.Sequential(
            conv_layer(304, 32),
            conv_layer(32, 64))

        self.conv_4_2 = nn.Sequential(
            conv_layer(52, 12),
            conv_layer(12, 12)
        )

    def _transformer(self, DMT1_yl, DMT1_yh):
        list_tensor = []
        for i in range(3):
            list_tensor.append(DMT1_yh[0][:, :, i, :, :])
        list_tensor.append(DMT1_yl)
        return torch.cat(list_tensor, 1)

    def _Itransformer(self, out):
        yh = []
        C = int(out.shape[1] / 4)
        y = out.reshape((out.shape[0], C, 4, out.shape[-2], out.shape[-1]))
        yl = y[:, :, 0].contiguous()
        yh.append(y[:, :, 1:].contiguous())
        return yl, yh

    def forward(self, x):
        x_2, y_2 = self.DWT(x)
        low2x = self._transformer(x_2, y_2)
        x_4, y_4 = self.DWT(low2x)
        low4x = self._transformer(x_4, y_4)
        x_8, y_8 = self.DWT(low4x)
        low8x = self._transformer(x_8, y_8)

        low8x_before = self.conv8x(low8x)

        out = []
        for i in range(9):
            if i == 0:
                input = low8x_before
            hgm3 = self.HGM_3[i](input)
            out.append(hgm3)
            input = hgm3

        low8x_after = torch.cat(out, dim=1)

        low4x_before = self.conv4x(low4x)

        out = []
        for j in range(6):
            if j == 0:
                input = low4x_before
            hgm2 = self.HGM_2[j](input)
            out.append(hgm2)
            input = hgm2

        low4x_after = torch.cat(out, dim=1)

        low2x_before = self.conv2x(low2x)

        out = []
        for k in range(3):
            if k == 0:
                input = low2x_before
            hgm1 = self.HGM_1[k](input)
            out.append(hgm1)
            input = hgm1
        low2x_after = torch.cat(out, dim=1)

        low8x_up2 = self.IDWT(self._Itransformer(self.downc8(low8x_after) + low8x_before))
        low4x_cat = torch.cat((low8x_up2, low4x_after), dim=1)

        low_8_4 = self.conv_8_4(low4x_cat)
        low4x_up2 = self.IDWT(self._Itransformer(low_8_4))
        low2x_cat = torch.cat((low4x_up2, low2x_after), dim=1)
        low_4_2 = self.conv_4_2(low2x_cat)
        low2x_up2 = self.IDWT(self._Itransformer(low_4_2))

        final = x - low2x_up2
        return torch.clamp(final, min=0.0, max=1.0)
