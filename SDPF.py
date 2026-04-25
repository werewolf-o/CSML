import torch
import torch.nn as nn
import torch.nn.functional as F
from work_02.shiyan_model.HoGEdge import HoGEdgeGateConv

class ShapeGuidedDiffusionPredictorFinal(nn.Module):
    def __init__(self, in_channels, mid_channels, kernel_size=3, ema_decay=0.9):
        super().__init__()
        self.mid_channels = mid_channels
        self.kernel_size = kernel_size
        self.k_sq = kernel_size * kernel_size
        self.ema_decay = ema_decay

        self.conv_predict = nn.Conv2d(
            in_channels,
            mid_channels * (1 + self.k_sq),
            kernel_size=1
        )
        self.register_buffer('ema_scale', torch.tensor(0.0))

    def forward(self, x):
        B, _, H, W = x.shape
        raw_out = self.conv_predict(x)

        # Split
        score_map = raw_out[:, :self.mid_channels, :, :]
        kernel_map = raw_out[:, self.mid_channels:, :, :].view(B, self.mid_channels, self.k_sq, H, W)

        # Global Scores for Decision
        global_scores = torch.sigmoid(F.adaptive_avg_pool2d(score_map, 1).flatten(1))

        # Norm Kernels
        norm_kernels = F.softmax(kernel_map, dim=2)

        # EMA & K calculation
        if self.training:
            current_scale = global_scores.mean(dim=1).mean()
            self.ema_scale = (self.ema_decay * self.ema_scale +
                              (1 - self.ema_decay) * current_scale.detach())
        scale = torch.clamp(self.ema_scale if self.training else global_scores.mean(), min=0.25, max=1.0)
        k = torch.ceil(self.mid_channels * scale).int().clamp(min=max(4, self.mid_channels // 4))

        # TopK
        _, indices = torch.topk(global_scores, k.item(), dim=1)
        mask = torch.zeros_like(global_scores).scatter(1, indices, 1.0).unsqueeze(-1).unsqueeze(-1)

        return mask, norm_kernels, k.item(), indices


class OptimizedDiffusionFusion(nn.Module):
    def __init__(self, in_channel, out_channel, edge_channels=36, mid_channels=64,
                 diffusion_kernel_size=3, diffusion_steps=2, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.mid_channels = mid_channels
        self.diffusion_steps = diffusion_steps
        self.k_size = diffusion_kernel_size
        self.pad = diffusion_kernel_size // 2

        self.EdgeConv = HoGEdgeGateConv(in_channel, in_channel)
        self.coo = nn.Conv2d(in_channel, edge_channels,1)
        self.predictor = ShapeGuidedDiffusionPredictorFinal(edge_channels, mid_channels,
                                                            kernel_size=diffusion_kernel_size)

        self.proj_depth = nn.Conv2d(in_channel, mid_channels, 1)
        self.proj_rgb = nn.Conv2d(out_channel, mid_channels, 1)

        self.out_conv = nn.Sequential(
            nn.Conv2d(mid_channels, out_channel, 1),
            norm_layer(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, gfusion1, gfusion2):
        B1, C1, H1, W1 = gfusion1.shape
        B2, C2, H2, W2 = gfusion2.shape

        # 1. Shape Prior
        edge_prior = self.coo(self.EdgeConv(gfusion1))  # [B, edge_channels, H, W]

        # 2. Predict Dynamics
        mask, kernels, k, indices = self.predictor(edge_prior)
        # indices: [B, k], kernels: [B, C_mid, k_sq, H, W]

        # 3. Align & Project
        if (H2, W2) != (H1, W1):
            gfusion2 = F.interpolate(gfusion2, size=(H1, W1), mode='bilinear', align_corners=False)
        feat_d = self.proj_depth(gfusion1)  # [B, mid_channels, H, W]
        feat_r = self.proj_rgb(gfusion2)  # [B, mid_channels, H, W]
        base_feat = F.relu(feat_d * feat_r)  # element-wise fusion

        # 4. Dynamic Channel Selection
        # indices: [B, k] -> select channels from base_feat
        B, C, H, W = base_feat.shape
        k_selected = indices.shape[1]

        # gather selected channels
        indices_exp = indices.view(B, k_selected, 1, 1).expand(B, k_selected, H, W)  # [B, k, H, W]
        active_feat = torch.gather(base_feat, 1, indices_exp)  # [B, k, H, W]

        # gather corresponding kernels
        # kernels: [B, mid_channels, k_sq, H, W] -> gather along dim=1
        _, C_mid, k_sq, Hk, Wk = kernels.shape
        indices_k = indices.view(B, k_selected, 1, 1, 1).expand(B, k_selected, k_sq, Hk, Wk)
        active_kernels = torch.gather(kernels, 1, indices_k)  # [B, k, k_sq, H, W]

        # 5. Message Passing (Diffusion)
        x_curr = active_feat  # [B, k, H, W]
        pad = self.k_size // 2

        for _ in range(self.diffusion_steps):
            x_pad = F.pad(x_curr, (pad, pad, pad, pad), mode='reflect')  # pad H&W
            x_unfold = F.unfold(x_pad, kernel_size=self.k_size)  # [B, k*k_size^2, H*W]
            x_unfold = x_unfold.view(B, k_selected, k_sq, H, W)  # [B, k, k_sq, H, W]

            # Weighted sum with dynamic kernels
            x_curr = (x_unfold * active_kernels).sum(dim=2)  # [B, k, H, W]

        # 6. Scatter back to full channel space
        out_full = torch.zeros_like(base_feat)
        out_full = out_full.scatter(1, indices_exp, x_curr)  # [B, C, H, W]

        # 7. Output conv
        out = self.out_conv(out_full)
        return out, edge_prior

if __name__ == "__main__":
    model = OptimizedDiffusionFusion(in_channel = 256, out_channel = 256)
    a = torch.rand(2, 256, 36, 64)
    b = torch.randn(2, 256, 18, 32)
    out = model(a,b)
    # out, out1, out2 = model(images)
    print(out.shape)