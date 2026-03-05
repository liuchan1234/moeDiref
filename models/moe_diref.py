"""
MoE-DiReF：双分支（结构保持 + 生成）+ REME 三阶段校正，端到端前向。
论文 Fig.1, III-A：结构分支 MKAE，生成分支 Rectified Flow，REME 约束扩散。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

from .mkae import MKAE
from .rectified_flow import RectifiedFlow, sample_ode_euler, sample_ode_heun
from .reme import REME


class LatentEncoder(nn.Module):
    """图像 -> 潜在 z，与 DoFe 同分辨率 (H/2, W/2)，通道数 latent_channels。"""

    def __init__(self, in_channels: int = 3, latent_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, latent_channels, 3, stride=2, padding=1),
            nn.GroupNorm(min(8, latent_channels), latent_channels),
            nn.SiLU(),
        )
        self.latent_channels = latent_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LatentDecoder(nn.Module):
    """潜在 z (H/4 若 encoder 两次 stride2 -> H/4) -> 图像。DoFe 为 H/2，encoder 两次 stride2 得 H/4。改为一次 stride2 得 H/2 与 DoFe 一致。"""

    def __init__(self, latent_channels: int = 64, out_channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent_channels, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, out_channels, 3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class MoEDiReF(nn.Module):
    """
    MoE-DiReF 主模型。
    - 结构分支：I_fus, M -> DoFe -> MKAE -> z_c, z_s
    - 生成分支：encoder -> z，Rectified Flow 采样（可选 REME 约束）-> z_0 -> decoder -> I_pred
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 64,
        dofe_out_channels: int = 64,
        mkae_base_channels: int = 32,
        mkae_out_channels: int = 64,
        num_experts: int = 4,
        rf_channels: int = 64,
        rf_time_embed_dim: int = 128,
        rf_blocks: int = 2,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.encoder = LatentEncoder(in_channels=in_channels, latent_channels=latent_channels)
        self.decoder = LatentDecoder(latent_channels=latent_channels, out_channels=in_channels)
        self.reme = REME(
            dofe_in_channels=in_channels,
            dofe_out_channels=dofe_out_channels,
            latent_channels=latent_channels,
            mkae_out_channels=mkae_out_channels,
        )
        self.mkae = MKAE(
            in_channels=dofe_out_channels,
            base_channels=mkae_base_channels,
            out_channels=mkae_out_channels,
            num_experts=num_experts,
        )
        self.rf = RectifiedFlow(
            latent_channels=latent_channels,
            channels=rf_channels,
            time_embed_dim=rf_time_embed_dim,
            num_blocks=rf_blocks,
        )

    def structure_branch(self, x: torch.Tensor, mask: torch.Tensor) -> tuple:
        """I_fus, M -> f_DoFe -> MKAE -> z_c, z_s。"""
        f_dofe = self.reme.forward_dofe(x, mask)
        z_c, z_s = self.mkae(f_dofe, noise_std=0.0)
        return z_c, z_s

    def sample_with_reme(
        self,
        z_1: torch.Tensor,
        z_c: torch.Tensor,
        z_s: torch.Tensor,
        num_steps: int = 10,
        solver: str = "euler",
    ) -> torch.Tensor:
        """从 z_1 沿 ODE 积分到 z_0，每步用 REME 约束。"""
        if solver == "heun":
            return self._ode_heun_with_reme(z_1, z_c, z_s, num_steps)
        return self._ode_euler_with_reme(z_1, z_c, z_s, num_steps)

    def _ode_euler_with_reme(
        self,
        z_1: torch.Tensor,
        z_c: torch.Tensor,
        z_s: torch.Tensor,
        num_steps: int,
    ) -> torch.Tensor:
        z = z_1
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.ones(z.shape[0], device=z.device, dtype=z.dtype) * (1.0 - i * dt)
            z = self.reme.constrain_latent(z, z_c, z_s)
            v = self.rf.v_net(z, t)
            z = z - v * dt
        return z

    def _ode_heun_with_reme(
        self,
        z_1: torch.Tensor,
        z_c: torch.Tensor,
        z_s: torch.Tensor,
        num_steps: int,
    ) -> torch.Tensor:
        z = z_1
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t_cur = 1.0 - i * dt
            t_next = 1.0 - (i + 1) * dt
            z = self.reme.constrain_latent(z, z_c, z_s)
            t_t = torch.ones(z.shape[0], device=z.device, dtype=z.dtype) * t_cur
            v_cur = self.rf.v_net(z, t_t)
            z_star = z - v_cur * dt
            z_star = self.reme.constrain_latent(z_star, z_c, z_s)
            t_n = torch.ones(z.shape[0], device=z.device, dtype=z.dtype) * t_next
            v_next = self.rf.v_net(z_star, t_n)
            z = z - 0.5 * (v_cur + v_next) * dt
        return z

    def forward(
        self,
        image_gt: torch.Tensor,
        image_fused: torch.Tensor,
        mask: torch.Tensor,
        return_loss_components: bool = True,
        num_steps: int = 10,
    ) -> Dict[str, Any]:
        """
        训练前向：计算 path loss 与生成 I_pred（用于外部 L_rec/L_adv/L_perc）。
        image_gt: I_gt [B,3,H,W]
        image_fused: I_fus [B,3,H,W]
        mask: M [B,1,H,W]
        num_steps: REME 约束 ODE 采样步数（训练时可减小以加速）。
        """
        z_c, z_s = self.structure_branch(image_fused, mask)
        z_0 = self.encoder(image_gt)
        z_1 = torch.randn_like(z_0, device=z_0.device, dtype=z_0.dtype)
        loss_path = self.rf.path_loss(z_0, z_1)
        z_0_pred = self.sample_with_reme(z_1, z_c, z_s, num_steps=num_steps, solver="euler")
        I_pred = self.decoder(z_0_pred)
        out = {"I_pred": I_pred, "z_0": z_0_pred, "z_c": z_c, "z_s": z_s}
        if return_loss_components:
            out["loss_path"] = loss_path
        return out

    @torch.no_grad()
    def infer(
        self,
        image_fused: torch.Tensor,
        mask: torch.Tensor,
        num_steps: int = 10,
        solver: str = "euler",
    ) -> torch.Tensor:
        """推理：I_fus, M -> I_pred。"""
        z_c, z_s = self.structure_branch(image_fused, mask)
        z_0_dummy = self.encoder(image_fused)
        z_1 = torch.randn_like(z_0_dummy, device=image_fused.device, dtype=image_fused.dtype)
        z_0 = self.sample_with_reme(z_1, z_c, z_s, num_steps=num_steps, solver=solver)
        return self.decoder(z_0)

    @torch.no_grad()
    def infer_with_intermediates(
        self,
        image_fused: torch.Tensor,
        mask: torch.Tensor,
        num_steps: int = 10,
        solver: str = "euler",
    ) -> Dict[str, Any]:
        """推理并返回中间量，用于图示：损坏输入、二值掩码、噪声潜码、恢复输出。"""
        z_c, z_s = self.structure_branch(image_fused, mask)
        z_0_dummy = self.encoder(image_fused)
        z_1 = torch.randn_like(z_0_dummy, device=image_fused.device, dtype=image_fused.dtype)
        z_0 = self.sample_with_reme(z_1, z_c, z_s, num_steps=num_steps, solver=solver)
        I_pred = self.decoder(z_0)
        # 噪声潜码可视化：用 decoder 解码 z_1 得到“噪声图像”
        noise_decoded = self.decoder(z_1)
        return {
            "I_pred": I_pred,
            "z_1": z_1,
            "z_0": z_0,
            "noise_decoded": noise_decoded,
        }
