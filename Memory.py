import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class Memory(nn.Module):
    def __init__(self, channel_dim, dilation, topk_spatial=2):
        super(Memory, self).__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(2 * channel_dim, channel_dim, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2)
        )
        self.depth_conv = nn.Sequential(
            nn.Conv2d(channel_dim, channel_dim, kernel_size=3, stride=1,
                      padding=dilation, groups=channel_dim, dilation=dilation),
            nn.LeakyReLU(0.2)
        )
        self.topk = topk_spatial

    def forward(self, image_feature, memory):
        """
        Args:
            image_feature: [B, C, H, W]
            memory: [M, C] (静态记忆库)
        Returns:
            updated_image: [B, C, H, W] 增强后的特征
        """
        # 1. 全局模式补偿 (Global Pattern Adjustment)
        I_G = F.adaptive_avg_pool2d(image_feature, (1, 1))  # [B,D,1,1]
        I_G = I_G.permute(0, 2, 3, 1)  # [B,1,1,D]


        # 计算分数 [B, M]
        score = torch.matmul(I_G, memory.t())  # [B, 1, 1, M]
        # print(score.shape)
        score = score.squeeze(1)
        score = score.squeeze(1)# [B, M]
        # print(score.shape)
        score_image = F.softmax(score, dim=1)

        # 获取记忆响应并融合
        I_G_flat = I_G.squeeze() # [B, D]
        memory_response = torch.matmul(score_image, memory)  # [B, D]
        memory_response = memory_response + I_G_flat
        memory_response = memory_response.unsqueeze(-1).unsqueeze(-1)  # [B, D, 1, 1]



        global_compensation = torch.sigmoid(memory_response) * image_feature

        # 2. 空间上下文细化 (Spatial Context Refinement)
        B, D, H, W = image_feature.shape
        M, _ = memory.shape

        memory_kernels = memory.detach().clone().view(M, D, 1, 1)
        score_maps = F.conv2d(image_feature, weight=memory_kernels)  # [B,M,H,W]

        score_maps_flat = score_maps.view(B, M, -1)  # [B,M,HW]
        topk_scores, topk_indices = torch.topk(score_maps_flat, k=self.topk, dim=1)
        attn_weights = F.softmax(topk_scores, dim=1)  # [B,K,HW]

        idx_flat = topk_indices.permute(0, 2, 1).reshape(B, -1)  # [B,HW*K]
        keys_exp = memory.unsqueeze(0).expand(B, -1, -1)  # [B,M,D]

        gathered_keys = torch.gather(
            keys_exp, dim=1,
            index=idx_flat.unsqueeze(-1).expand(-1, -1, memory.size(1))
        )  # [B,HW*K,D]

        gathered_keys = gathered_keys.view(B, H * W, self.topk, -1).permute(0, 2, 1, 3)  # [B,K,HW,D]
        attn_weights = attn_weights.unsqueeze(-1)  # [B,K,HW,1]

        memory_feat_flat = torch.sum(attn_weights * gathered_keys, dim=1)  # [B,HW,D]
        memory_feat = memory_feat_flat.transpose(1, 2).reshape(B, D, H, W)

        # print(global_compensation.shape)
        # print(memory_feat.shape)
        # 3. 融合输出
        fused = torch.cat([global_compensation, memory_feat], dim=1)
        updated_image = self.fusion(fused)
        updated_image = self.depth_conv(updated_image)

        return updated_image


# ==========================================
# 2. 专家封装与路由逻辑 (简化版)
# ==========================================

class SingleLayerMemoryExpert(nn.Module):
    """单个层级的记忆专家 (纯推理)"""

    def __init__(self, channels, dilation, top_k_spatial=2):
        super().__init__()
        self.memory_module = Memory(channel_dim=channels, dilation=dilation, topk_spatial=top_k_spatial)

    def forward(self, x, memory_bank):
        """
        Args:
            x: [B, C, H, W]
            memory_bank: [M, C]
        Returns:
            out: [B, C, H, W]
        """
        out = self.memory_module(x, memory_bank)
        return out


class CAD_Memory_Router(nn.Module):
    """CAD 风格路由：首尾必选 + 中间 Top-K"""

    def __init__(self, in_channels, num_layers, top_k_middle=1):
        super().__init__()
        self.num_layers = num_layers
        self.top_k_middle = top_k_middle

        self.prompt_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // 2),
            nn.GELU()
        )

        self.router = nn.Sequential(
            nn.Linear((in_channels // 2) * num_layers, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, num_layers),
        )

    def forward(self, features: List[torch.Tensor],promts) -> torch.Tensor:
        B = features[0].shape[0]
        device = features[0].device

        prompts = []
        for feat in promts:
            p = self.prompt_proj(feat)
            prompts.append(p)

        concat_prompts = torch.cat(prompts, dim=1)
        scores = self.router(concat_prompts)
        scores = F.sigmoid(scores)

        mask = torch.zeros_like(scores, dtype=torch.bool, device=device)
        mask[:, 0] = True  # 最浅层必选
        mask[:, -1] = True  # 最深层必选

        if self.num_layers > 2:
            middle_scores = scores[:, 1:-1]
            k = min(self.top_k_middle, middle_scores.shape[1])
            if k > 0:
                _, top_idx = torch.topk(middle_scores, k, dim=1)
                original_idx = top_idx + 1
                mask.scatter_(1, original_idx, True)

        weighted_scores = scores * mask.float()
        sum_weights = weighted_scores.sum(dim=1, keepdim=True) + 1e-6
        normalized_weights = weighted_scores / sum_weights

        return normalized_weights, concat_prompts