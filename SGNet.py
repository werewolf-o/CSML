import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone.Dformer.DFormer2 import DFormer_Small
from work_02.shiyan_model.SDPF import OptimizedDiffusionFusion
from work_02.shiyan_model.HoGEdge import HoGEdgeGateConv

class BasicConv2d(nn.Module):  # 很多模块的使用卷积层都是以其为基础，论文中的BConvN
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class FAM(nn.Module):
    def __init__(self, in_channel_left, in_channel_down, norm_layer=nn.BatchNorm2d):
        super(FAM, self).__init__()
        # self.conv_d1 = nn.Conv2d(in_channel_down, 256, kernel_size=3, stride=1, padding=1)
        # self.conv_l = nn.Conv2d(in_channel_left, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256 * 2, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = norm_layer(256)

    def forward(self, left, down):
        # down_mask = self.conv_d1(down)
        # left_mask = self.conv_l(left)
        down_mask = down
        left_mask = left
        if down.size()[2:] != left_mask.size()[2:]:
            down_ = F.interpolate(down, size=left.size()[2:], mode='bilinear')
            z1 = F.relu(left_mask * down_, inplace=True)
        else:
            z1 = F.relu(left_mask * down, inplace=True)

        if down_mask.size()[2:] != left.size()[2:]:
            down_mask = F.interpolate(down_mask, size=left.size()[2:], mode='bilinear')

        z2 = F.relu(down_mask * left, inplace=True)

        out = torch.cat((z1, z2), dim=1)
        return F.relu(self.bn3(self.conv3(out)), inplace=True)

class Fusion(nn.Module):
    def __init__(self, in_channel, norm_layer=nn.BatchNorm2d):
        super(Fusion, self).__init__()
        self.conv0 = nn.Conv2d(in_channel * 2, 256, 3, 1, 1)
        self.bn0 = norm_layer(256)

    def forward(self, x1, x2, alpha, beta):
        out1 = alpha * x1 + beta * (1.0 - alpha) * x2
        out2 = x1 * x2
        out = torch.cat((out1, out2), dim=1)
        out = F.relu(self.bn0(self.conv0(out)), inplace=True)

        return out

class side_fusion(nn.Module):
    def __init__(self, in_channel, norm_layer=nn.BatchNorm2d):
        super(side_fusion, self).__init__()
        self.conv0 = nn.Conv2d(in_channel * 2, 256, 3, 1, 1)
        self.bn0 = norm_layer(256)

    def forward(self, sideout1, sideout2):

        out = torch.cat((sideout1, sideout2), dim=1)
        out = F.relu(self.bn0(self.conv0(out)), inplace=True)

        return out

class SGNet(nn.Module):
    def __init__(self):
        super(SGNet, self).__init__()

        self.head = nn.Conv2d(64, 3, 1)
        self.label_r = DFormer_Small()
        self.label_r._init_weights("/home/yph/lx/lx/主要模型/PotCrackSeg-main-2/backbone/Dformer/DFormer_Small.pth.tar")

        # self.decoder02 = Decoder(dim=[64,128,256,512],out_c=256, num_classes=3)

        channels = [64, 128, 256, 512]

        # self.decoder02 = Decoder(dim=[64,128,256,512],out_c=256, num_classes=3)

        # low-level & high-level
        self.fam54_1 = FAM(256, 256)
        self.fam43_1 = FAM(256, 256)
        self.fam32_1 = FAM(256, 256)
        self.fam54_2 = FAM(256, 256)
        self.fam43_2 = FAM(256, 256)
        self.fam32_2 = FAM(256, 256)
        # fusion, TBD
        self.fusion = Fusion(256)
        self.sidefusion1 = side_fusion(256)
        self.sidefusion2 = side_fusion(256)

        # self.gfusion1 = global_fusion(256, 256)
        # self.gfusion2 = global_fusion(256, 256)
        # self.gfusion3 = global_fusion(256, 256)
        # self.gfusion4 = global_fusion(256, 256)

        self.gfusion1 = OptimizedDiffusionFusion(in_channel = 256, out_channel = 256)
        self.gfusion2 = OptimizedDiffusionFusion(in_channel =256, out_channel = 256)
        self.gfusion3 = OptimizedDiffusionFusion(in_channel =256, out_channel = 256)
        self.gfusion4 = OptimizedDiffusionFusion(in_channel =256, out_channel = 256)

        self.sigmoid = nn.Sigmoid()
        self.linear_out = nn.Conv2d(256, 3, kernel_size=3, stride=1, padding=1)
        self.gap1 = nn.AdaptiveAvgPool2d(1)
        self.gap2 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels[-1], 512),
            ##nn.Dropout(p=0.3),
            nn.ReLU(True),
            nn.Linear(512, 256 + 1),
            nn.Sigmoid(),
        )

        # decoder01
        self.reduce1 = nn.Conv2d(64, 256, 1)
        self.reduce2 = nn.Conv2d(128, 256, 1)
        self.reduce3 = nn.Conv2d(256, 256, 1)
        self.reduce4 = nn.Conv2d(512, 256, 1)

        self.reduce1_d = nn.Conv2d(32, 256, 1)
        self.reduce2_d = nn.Conv2d(64, 256, 1)
        self.reduce3_d = nn.Conv2d(128, 256, 1)
        self.reduce4_d = nn.Conv2d(256, 256, 1)


    def encode_decode(self, x,x_e):
        raw_size = x.size()[2:]
        bz = x.shape[0]

        # print(x.shape)
        # print(x_e.shape)

        if x_e is None:
            x_e = x
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        if len(x_e.shape) == 3:
            x_e = x_e.unsqueeze(0)


        x_e = x_e[:, 0, :, :].unsqueeze(1)
        # print(x_e.shape)

        rd0, d0 = self.label_r.blk0(x, x_e)
        rd1, d1 = self.label_r.blk1(rd0, d0)
        rd2, d2 = self.label_r.blk2(rd1, d1)
        rd3, d3 = self.label_r.blk3(rd2, d2)

        enc2_1 = self.reduce1(rd0)
        enc3_1 = self.reduce2(rd1)
        enc4_1 = self.reduce3(rd2)
        enc5_1 = self.reduce4(rd3)

        enc2_2 = self.reduce1_d(d0)
        enc3_2 = self.reduce2_d(d1)
        enc4_2 = self.reduce3_d(d2)
        enc5_2 = self.reduce4_d(d3)

        rgb_gap = self.gap1(enc5_1)
        rgb_gap = rgb_gap.view(bz, -1)
        depth_gap = self.gap2(enc5_2)
        depth_gap = depth_gap.view(bz, -1)
        feat = torch.cat((rgb_gap, depth_gap), dim=1)
        # print(feat.shape)
        feat = self.fc(feat)

        gate = feat[:, -1].view(bz, 1, 1, 1)

        alpha = feat[:, :256]
        alpha = alpha.view(bz, 256, 1, 1)

        out4_1 = self.fam54_1(enc4_1, enc5_1)
        out4_2 = self.fam54_2(enc4_2, enc5_2)
        side_fusion4 = self.sidefusion1(out4_1, out4_2)
        de3_1, edge_prior3_1 = self.gfusion1(enc3_1, side_fusion4)
        de3_2, edge_prior3_2 = self.gfusion2(enc3_2, side_fusion4)

        out3_1 = self.fam43_1(de3_1, out4_1)
        out3_2 = self.fam43_2(de3_2, out4_2)
        side_fusion3 = self.sidefusion2(out3_1, out3_2)
        de2_1, edge_prior2_1 = self.gfusion3(enc2_1, side_fusion3)
        de2_2, edge_prior2_2 = self.gfusion4(enc2_2, side_fusion3)
        # F ronghe
        out2_1 = self.fam32_1(de2_1, out3_1)
        out2_2 = self.fam32_2(de2_2, out3_2)

        # print(edge_prior3_1.shape)
        # print(edge_prior3_2.shape)
        # print(edge_prior2_1.shape)
        # print(edge_prior2_2.shape)

        edge_prior =  F.interpolate((edge_prior3_1 + edge_prior3_2), edge_prior2_1.size()[2:], mode='bilinear') + (edge_prior2_1 + edge_prior2_2)
        # print(edge_prior.shape)

        # final fusion
        out = self.fusion(out2_1, out2_2, alpha, gate)
        out = F.interpolate(self.linear_out(out), size=raw_size, mode='bilinear', )

        out1 = F.interpolate(self.linear_out(side_fusion4), size=raw_size, mode='bilinear')
        out2 = F.interpolate(self.linear_out(side_fusion3), size=raw_size, mode='bilinear')

        return enc2_1, enc3_1, enc4_1, enc5_1, out, out1, out2, edge_prior

    def forward(self, input):

        rgb = input[:, :3]
        modal_x = input[:, 3:]
        # modal_x = torch.cat((modal_x, modal_x, modal_x), dim=1)

        enc2_1, enc3_1, enc4_1, enc5_1, out, out1, out2, edge_prior = self.encode_decode(rgb,modal_x)

        return enc2_1, enc3_1, enc4_1, enc5_1, out, out1, out2, edge_prior


if __name__ == "__main__":
    model = LXNet2().eval()
    with torch.no_grad():
        a = torch.rand(2, 3, 288, 512)
        b = torch.randn(2, 1, 288, 512)
        images = torch.cat([a, b], dim=1)
        enc2_1, enc3_1, enc4_1, enc5_1, out, out1, out2, edge_prior = model(images)
        print(enc2_1.shape)
        print(enc3_1.shape)
        print(enc4_1.shape)
        print(enc5_1.shape)
        print(out.shape)
        print(out1.shape)
        print(out2.shape)

    # from util.util import compute_speed
    # from ptflops import get_model_complexity_info
    # with torch.cuda.device(0):
    #     net = LXNet2().eval()
    #     flops, params = get_model_complexity_info(net, (4, 288, 512), as_strings=True, print_per_layer_stat=False)
    #     print('Flops: ' + flops)
    #     print('Params: ' + params)
    # #
    # compute_speed(net, input_size=(1, 4, 288, 512), iteration=500)
