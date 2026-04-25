import torch
import torch.nn as nn
import torch.nn.functional as F
from work_02.shiyan_model.Adapter import DynamicFilter
from work_02.shiyan_model.Memory import SingleLayerMemoryExpert,CAD_Memory_Router

class BasicConv2d(nn.Module):    #很多模块的使用卷积层都是以其为基础，论文中的BConvN
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
        self.conv_d1 = nn.Conv2d(in_channel_down, 256, kernel_size=3, stride=1, padding=1)
        self.conv_l = nn.Conv2d(in_channel_left, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256*2, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = norm_layer(256)

    def forward(self, left, down):
        down_mask = self.conv_d1(down)
        left_mask = self.conv_l(left)
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

class side_fusion(nn.Module):
    def __init__(self, in_channel, norm_layer =nn.BatchNorm2d):
        super(side_fusion,self).__init__()
        self.conv0 = nn.Conv2d(in_channel*2, 256, 3, 1, 1)
        self.bn0 = norm_layer(256)
    def forward(self,sideout1,sideout2):
        out = torch.cat((sideout1,sideout2),dim=1)
        out = F.relu(self.bn0(self.conv0(out)),inplace=True)

        return out

class GlobalFusion_Single(nn.Module):
    def __init__(self, in_channel, norm_layer=nn.BatchNorm2d):
        super(GlobalFusion_Single, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 256, 3, 1, 1)
        self.conv2 = nn.Conv2d(256,       256, 3, 1, 1)
        self.conv3 = nn.Conv2d(256 * 2,   256, 3, 1, 1)
        self.bn0 = norm_layer(256)

    def forward(self, x_enc, x_side):
        x_enc  = self.conv1(x_enc)
        x_side = self.conv2(x_side)
        if x_side.size()[2:] != x_enc.size()[2:]:
            x_side = F.interpolate(x_side, x_enc.size()[2:], mode='bilinear')
        fused = x_enc * x_side
        out = torch.cat((x_enc, fused), dim=1)
        return F.relu(self.bn0(self.conv3(out)), inplace=True)

class Fusion_Single(nn.Module):
    def __init__(self, in_channel, norm_layer=nn.BatchNorm2d):
        super(Fusion_Single, self).__init__()
        self.conv0 = nn.Conv2d(in_channel * 2, 256, 3, 1, 1)
        # self.conv0 = nn.Conv2d(in_channel, 256, 3, 1, 1)
        self.bn0 = norm_layer(256)

    def forward(self, x, alpha, beta):
        # alpha: (B, C, 1, 1), beta: (B,1,1,1)
        out1 = alpha * x
        out2 = beta * x * x
        out  = torch.cat((out1, out2), dim=1)
        return F.relu(self.bn0(self.conv0(out)), inplace=True)


class OptimizedAdapter(nn.Module):
    def __init__(self, blk, dim=None, num_classes=3):
        super().__init__()
        self.block = blk
        if dim is None:
            dim = blk.dwconv.out_channels
        # Adapter2 DynamicFilter：forward → (identity+x, prompt)
        # prompt shape: (B, H, W, med_channels=128)
        self.mona = DynamicFilter(dim=dim, med_channels=128)
        self.last_prompt = None  # 缓存槽，每次 forward 后更新

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            out: (B, C, H, W)  —— 返回签名与原始 block 完全一致
        Side-effect:
            self.last_prompt ← (B, H, W, 128)  供外部 encode_decode 读取
        """
        # (B,C,H,W) → (B,H,W,C)
        x_permuted = x.permute(0, 2, 3, 1)

        # Adapter2 返回 (enhanced_feat, prompt)
        x_enhanced, prompt = self.mona(x_permuted)  # prompt: (B, H, W, 128)

        # 缓存 prompt（保留梯度，参与训练）
        self.last_prompt = prompt

        # (B,H,W,C) → (B,C,H,W)，交给原始 ConvNeXt block
        x_enhanced = x_enhanced.permute(0, 3, 1, 2).contiguous()
        return self.block(x_enhanced)


class Segment(nn.Module):
    def __init__(
            self,
            dinov3_weight_path='/home/yph/lx/lx/主要模型/PotCrackSeg-main-2/dino/'
                               'dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth',
            dinov3_local_path='/home/yph/lx/lx/主要模型/PotCrackSeg-main-2/dino/dinov3',
            cfg=None,
            aux_layers=True,
    ):
        super(Segment, self).__init__()
        self.cfg = cfg
        self.aux_layers = aux_layers
        channels = 256
        num_experts = 4
        prompt_channels = 128  # Adapter2 DynamicFilter 的 med_channels

        # ── Backbone ─────────────────────────────────────────────────────────
        self.dino1 = torch.hub.load(
            repo_or_dir=dinov3_local_path,
            model='dinov3_convnext_large',
            source='local',
            pretrained=False,
            trust_repo=True,
        )
        if dinov3_weight_path:
            checkpoint = torch.load(dinov3_weight_path, map_location='cpu')
            self.dino1.load_state_dict(checkpoint, strict=True)
            print('✓ Local weights successfully loaded')

        for param in self.dino1.parameters():
            param.requires_grad = False

        # ── 载入适配器（Adapter2）────────────────────────────────────────────
        adapted_stages = []
        for stage in self.dino1.stages:
            adapted_blocks = []
            for block in stage:
                adapted_blocks.append(OptimizedAdapter(block, num_classes=3))
            adapted_stages.append(nn.Sequential(*adapted_blocks))
        self.dino1.stages = nn.ModuleList(adapted_stages)

        # ── Prompt 降维：128 → 256，每个 stage 独立一个 1×1 Conv ─────────────
        # 输入: (B, 128, H_i, W_i)  输出: (B, 256, H_i, W_i)
        self.prompt_reduce = nn.ModuleList([
            nn.Conv2d(prompt_channels, channels, kernel_size=1, bias=False)
            for _ in range(num_experts)
        ])

        # ── 并行专家池 ────────────────────────────────────────────────────────
        dilations = [1, 3, 5, 7]
        self.expert_pool = nn.ModuleList([
            SingleLayerMemoryExpert(channels=channels, dilation=d)
            for d in dilations
        ])

        # ── 记忆库 ────────────────────────────────────────────────────────────
        self.memory_banks = nn.ParameterList([
            nn.Parameter(torch.randn(10, channels), requires_grad=True)
            for _ in range(num_experts)
        ])

        # ── CAD 路由器（双输入版）────────────────────────────────────────────
        #   routing_weights ← reduced_feats    (语义路由，逻辑不变)
        #   F_prompt        ← reduced_prompts  (Adapter2 频域 prompt)
        self.cad_router = CAD_Memory_Router(
            in_channels=channels,
            num_layers=num_experts,
            top_k_middle=1,
        )

        # ── 解码 ─────────────────────────────────────────────────────────────
        self.fam54 = FAM(256, 256)
        self.fam43 = FAM(256, 256)
        self.fam32 = FAM(256, 256)

        self.gfusion3 = GlobalFusion_Single(256)
        self.gfusion2 = GlobalFusion_Single(256)
        self.gfusion1 = GlobalFusion_Single(256)

        self.fusion = Fusion_Single(256)
        self.sigmoid = nn.Sigmoid()
        self.linear_out = nn.Conv2d(256, 3, kernel_size=3, stride=1, padding=1)
        self.gap1 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 256 + 1),
            nn.Sigmoid(),
        )
        self.reduce1 = nn.Conv2d(192, 256, 1)
        self.reduce2 = nn.Conv2d(384, 256, 1)
        self.reduce3 = nn.Conv2d(768, 256, 1)
        self.reduce4 = nn.Conv2d(1536, 256, 1)

    def encode_decode(self, rgb):
        raw_size = rgb.size()[2:]
        bz = rgb.shape[0]

        # ── Step 1：调用 _get_intermediate_layers（与原始代码完全一致）────────
        # 内部顺序经过各 stage 的 OptimizedAdapter.forward，
        # 每个 block 执行时将自身的 prompt 写入 self.last_prompt
        x = self.dino1._get_intermediate_layers(rgb, n=[0, 1, 2, 3])

        # ── Step 2：特征降维（与原始代码完全一致）────────────────────────────
        enc2 = self.reduce1(x[0][1])  # (B, 256, H/4,  W/4 )
        enc3 = self.reduce2(x[1][1])  # (B, 256, H/8,  W/8 )
        enc4 = self.reduce3(x[2][1])  # (B, 256, H/16, W/16)
        enc5 = self.reduce4(x[3][1])  # (B, 256, H/32, W/32)
        reduced_feats = [enc2, enc3, enc4, enc5]

        # ── Step 3：读取各 stage 最后一个 block 缓存的 prompt ─────────────────
        # dino1.stages[i] 是 nn.Sequential，[-1] 取最后一个 OptimizedAdapter
        # last_prompt: (B, H_i, W_i, 128)
        raw_prompts = [
            self.dino1.stages[i][-1].last_prompt
            for i in range(4)
        ]

        # ── Step 4：Prompt 降维 128 → 256 ─────────────────────────────────────
        # (B, H_i, W_i, 128) → permute → (B, 128, H_i, W_i) → Conv1×1 → (B, 256, H_i, W_i)
        reduced_prompts = [
            self.prompt_reduce[i](raw_prompts[i].permute(0, 3, 1, 2).contiguous())
            for i in range(4)
        ]

        # ── Step 5：路由决策 ──────────────────────────────────────────────────
        # routing_weights: [B, 4]    由 reduced_feats   决定（首尾必选 Top-K 路由）
        # F_prompt:        [B, 256]  由 reduced_prompts 决定（Adapter2 频域 prompt）
        routing_weights, F_prompt = self.cad_router(reduced_feats, reduced_prompts)

        # ── Step 6：并行专家处理与融合 ────────────────────────────────────────
        aggregated_feat = torch.zeros_like(reduced_feats[0])
        for idx, (expert, mem_bank, inp_feat) in enumerate(
                zip(self.expert_pool, self.memory_banks, reduced_feats)
        ):
            w = routing_weights[:, idx].view(bz, 1, 1, 1)
            enhanced_feat = expert(inp_feat, mem_bank)
            enhanced_feat = F.interpolate(enhanced_feat, aggregated_feat.size()[2:], mode='bilinear')
            aggregated_feat = aggregated_feat + enhanced_feat * w

        gap = self.gap1(aggregated_feat).view(bz, -1)
        feat = self.fc(gap)
        gate = feat[:, -1].view(bz, 1, 1, 1)
        alpha = feat[:, :256].view(bz, 256, 1, 1)

        # ── Step 7：解码（与原始代码完全一致）────────────────────────────────
        out4 = self.fam54(enc4, enc5)  # H/16
        de3 = self.gfusion1(enc3, out4)  # H/8
        out3 = self.fam43(de3, out4)  # H/8
        de2 = self.gfusion2(enc2, out3)  # H/4
        out2 = self.fam32(de2, out3)  # H/4

        outt = self.fusion(out2, alpha, gate)
        out1 = F.interpolate(self.linear_out(out4), size=raw_size, mode='bilinear')
        out2 = F.interpolate(self.linear_out(out3), size=raw_size, mode='bilinear')
        out = F.interpolate(self.linear_out(outt), size=raw_size, mode='bilinear')

        return enc2, enc3, enc4, enc5, out, out1, out2, F_prompt

    def forward(self, input):
        rgb = input[:, :3]
        modal_x = input[:, 3:]
        modal_x = torch.cat((modal_x, modal_x, modal_x), dim=1)

        enc2, enc3, enc4, enc5, out, out1, out2, F_prompt = self.encode_decode(rgb)
        return enc2, enc3, enc4, enc5, out, out1, out2, F_prompt


    def forward(self, input):

        rgb = input[:, :3]
        modal_x = input[:, 3:]
        modal_x = torch.cat((modal_x, modal_x, modal_x), dim=1)

        enc2, enc3, enc4, enc5, out,out1,out2, F_prompt= self.encode_decode(rgb)

        return enc2, enc3, enc4, enc5, out,out1,out2, F_prompt

if __name__ == "__main__":
    model = Segment().eval()
    with torch.no_grad():
        a = torch.rand(1, 3, 288, 512)
        b = torch.randn(1, 1, 288, 512)
        images = torch.cat([a, b], dim=1)
        enc2, enc3, enc4, enc5, out,out1,out2, F_prompt = model(images)
        print(out.shape)

    # from util.util import compute_speed
    # from ptflops import get_model_complexity_info
    # with torch.cuda.device(0):
    #     net = Segment().eval()
    #     flops, params = get_model_complexity_info(net, (4, 288, 512), as_strings=True, print_per_layer_stat=False)
    #     print('Flops: ' + flops)
    #     print('Params: ' + params)
    # #
    # compute_speed(net, input_size=(1, 4, 288, 512), iteration=500)
