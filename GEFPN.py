import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Head, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, out_channels, 1, 1, 0)
        )

    def forward(self, x):
        return self.block(x)


class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):

        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        out_normal = self.conv(x)

        kernel_diff1 = self.conv.weight.sum(2).sum(2)
        kernel_diff2 = kernel_diff1[:, :, None, None]
        out_diff = F.conv2d(input=x, weight=kernel_diff2, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

        return out_normal - out_diff


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample):
        super(ResidualBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),

            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        residual = x
        x = self.body(x)

        if self.downsample:
            residual = self.downsample(residual)

        out = F.relu(x+residual, True)
        return out

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.ReLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=1, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=s,padding=p, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        result = self.act(self.bn(self.conv(x)))
        return result

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3),d=(3,3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * 0.25)  # hidden channels
        #self.cv1 = Conv2d_cd(c1, c_, kernel_size=k[0], stride=1, padding=d[1], dilation=d[1])
        self.cv1 = Conv(c1, c_, 1, 1, 0)
        #self.cv3 = Conv(c_, c_, k[1], s=1, p=d[1],g=g,d=d[1])
        self.cv2 = Conv(c_, c2, k[1], s=1, p=d[1], g=g, d=d[1])
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return self.cv2(self.cv1(x))


class DCSPM(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.25):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()

        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1, p=0)
        self.cv2 = Conv(c1, 2 * self.c, 1, 1, p=0)
        self.cv3 = Conv(c1, 2 * self.c, 1, 1, p=0)

        self.bn2 = nn.BatchNorm2d(c2)
        self.conv1 = nn.Conv2d(6 * self.c, c2, kernel_size=1)
        self.conv2 = nn.Conv2d(5 * self.c, c2, kernel_size=1)
        self.conv3 = nn.Conv2d(3 * self.c, c2, kernel_size=1)

        self.m1 = nn.ModuleList([Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), d=(3, 3)),
                                 Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), d=(5, 5)),
                                 Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), d=(7, 7))])


        self.act = nn.ReLU()

    def forward(self, x, x_grad):
        """Forward pass of a YOLOv5 CSPDarknet backbone layer."""
        y1 = list(self.cv1(x).chunk(2, 1))
        y1.extend(m(y1[-1]) for m in self.m1)

        result1 = self.conv1(torch.cat([y1[1], y1[2], y1[3], y1[4], y1[0], x_grad], 1))
        result = self.act(result1 + x)

        return result



class PAFM(nn.Module):
    def __init__(self, channels_high, channels_low):
        super(PAFM, self).__init__()

        self.dconv = nn.ConvTranspose2d(in_channels=channels_high, out_channels=channels_low, kernel_size=3, stride=2, padding=1)

        self.GAP = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels_low, channels_low // 4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels_low // 4),
            nn.ReLU(True),
            nn.Conv2d(channels_low // 4, channels_low, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels_low)
        )

        self.AAP = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(channels_low, channels_low // 4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels_low // 4),
            nn.ReLU(True),
            nn.Conv2d(channels_low // 4, channels_low, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels_low)
        )
        self.active = nn.Sigmoid()
        self.conv = nn.Sequential(
            nn.Conv2d(channels_low, channels_low // 4, 1),
            nn.BatchNorm2d(channels_low // 4),
            nn.ReLU(True),
            nn.Conv2d(channels_low // 4, channels_low, 1),
            nn.BatchNorm2d(channels_low),
            nn.Sigmoid()

        )


    def forward(self, x_high, x_low):
        _, _, h, w = x_low.shape
        x_high = self.dconv(x_high, output_size=x_low.size())
        x_sum = x_high + x_low
        x_conv = self.conv(x_sum)

        tmp1 = self.GAP(x_high)
        tmp2 = self.AAP(x_high)
        p_x_high = self.active((self.GAP(x_high) + self.AAP(x_high)))
        p_x_high =  F.interpolate(p_x_high, scale_factor=h // 4, mode='nearest')
        output = ((x_conv) * x_sum) * p_x_high
        # output = (x_low + x_high) * p_x_high #* x_low_conv
        return output


class GEFPN(nn.Module):
    def __init__(self, layer_blocks=[4, 4, 4], channels=[16, 16, 32, 64]):
        super(GEFPN, self).__init__()

        stem_width = channels[0]
        self.stem1 = nn.Sequential(
            nn.Conv2d(1, stem_width, 3, 1, 1, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.ReLU(True)
        )

        self.stem2 = nn.Sequential(
            nn.Conv2d(stem_width, channels[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(True)
        )

        self.conv1 =nn.Sequential(
            nn.Conv2d(1, channels[0],  1 ),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(True),
            nn.Conv2d(channels[0], channels[0]//4, 3,1,1),
            nn.BatchNorm2d(channels[0]//4),
            nn.ReLU(True),
        )

        self.conv2 =nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(1, channels[2],  1 ),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(True),
            nn.Conv2d(channels[2], channels[2]//4, 3,1,1),
            nn.BatchNorm2d(channels[2]//4),
            nn.ReLU(True),
        )

        self.conv3 =nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=4, padding=0),
            nn.Conv2d(1, channels[3],  1 ),
            nn.BatchNorm2d(channels[3]),
            nn.ReLU(True),
            nn.Conv2d(channels[3], channels[3]//4, 3,1,1),
            nn.BatchNorm2d(channels[3]//4),
            nn.ReLU(True),
        )

        self.downsample = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_aux = nn.Sequential(
            #nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True),
            nn.Conv2d(channels[0], channels[0]//4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels[0]//4),
            nn.ReLU(True),
            nn.Conv2d(channels[0]//4, channels[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(True),
            #nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # nn.Conv2d(channels[0], channels[0]//4, 3, 1, 1, bias=False),
            # nn.BatchNorm2d(channels[0]//4),
            # nn.ReLU(True),
            # nn.Conv2d(channels[0] //4, channels[0], 3, 1, 1, bias=False),
            # nn.BatchNorm2d(channels[0]),
            # nn.ReLU(True),
        )
        self.head_aux = Head(channels[1] , 1)

        #self.firstloc = M2AM()
        #self.GF_layer = GF_Conv(16, 16, 1)

        self.layer1 = self._make_layer(block=ResidualBlock, block_num=layer_blocks[0],
                                       in_channels=channels[0], out_channels=channels[1], stride=1)
        self.layer2 = self._make_layer(block=ResidualBlock, block_num=layer_blocks[1],
                                       in_channels=channels[1], out_channels=channels[2], stride=2)
        self.layer3 = self._make_layer(block=ResidualBlock, block_num=layer_blocks[2],
                                       in_channels=channels[2], out_channels=channels[3], stride=2)

        self.refineloc0 = DCSPM(channels[0], channels[0])

        self.refineloc1 = DCSPM(channels[1], channels[1])
        self.refineloc2 = DCSPM(channels[2], channels[2])
        self.refineloc3 =DCSPM(channels[3], channels[3])

        self.fuse23 = PAFM(channels_high=channels[3], channels_low=channels[2])
        self.fuse12 = PAFM(channels_high=channels[2], channels_low=channels[1])

        self.head = Head(channels[1] , 1)
        self.sigmoid = nn.Sigmoid()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


    def forward(self, x, x_sobel):
        _, _, hei, wid = x.shape

        #mloc = self.firstloc(x)

        c0 = self.stem1(x)
        x_grad1 = self.conv1(x_sobel)
        x_grad2 = self.conv2(x_sobel)
        x_grad3 = self.conv3(x_sobel)

        c1 = self.layer1(c0)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        rc3 = self.refineloc3(c3, x_grad3)
        rc2 = self.refineloc2(c2, x_grad2)
        rc1 = self.refineloc1(c1, x_grad1)
        out1 = self.fuse23(rc3, rc2)
        out2 = self.fuse12(out1, rc1)


        pred = self.head(out2)

        return pred

    def _make_layer(self, block, block_num, in_channels, out_channels, stride):
        downsample = (in_channels != out_channels) or (stride != 1)
        layer = []
        layer.append(block(in_channels, out_channels, stride, downsample))
        for _ in range(block_num - 1):
            layer.append(block(out_channels, out_channels, 1, False))
        return nn.Sequential(*layer)

if __name__ == '__main__':
    from thop import profile
    import time
    net = GEFPN().cuda()
    net.eval()
    num_frames = 100  # 测试的帧数
    start_time = time.time()

    for _ in range(num_frames):
        # 生成或读取输入数据
        input_tensor = torch.rand(1, 1, 256, 256).cuda()  # 示例输入

        with torch.no_grad():  # 不计算梯度
            output = net(input_tensor, input_tensor)  # 执行推理

    end_time = time.time()
    total_time = end_time - start_time
    fps = num_frames / total_time
    print(f"Measured FPS: {fps:.2f}")

    dummy_input = torch.rand(1, 1, 256, 256).cuda()
    dummy_input_sobel = torch.rand(1, 1, 256, 256).cuda()
    FLOPs, params = profile(net, inputs=(dummy_input, dummy_input_sobel,))
    print('FLOPs=', str(FLOPs / 1000000.0) + '{}'.format('M'))
    print('params=', str(params / 1000000.0) + '{}'.format('M'))