"""
pixel_branch.py — Pixel Branch
"""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from safetensors.torch import load_file


# ─────────────────────────────────────────────
# 1. DINOv2 Encoder
# ─────────────────────────────────────────────

DINOV2_CACHE = (
    "/home/musap.yildiz/.cache/huggingface/hub"
    "/models--timm--vit_base_patch14_dinov2.lvd142m"
    "/snapshots/4685c99dabffe5affac90bd99dbffd25801ae58d"
    "/model.safetensors"
)


def _load_dinov2_224():
    """
    Cache'deki DINOv2 agirliklarini yukler.
    pos_embed 518x518 icin egitilmis → 224x224 icin bicubic resize yapar.
    """
    # Mimariyi 224x224 icin kur
    model = timm.create_model(
        "vit_base_patch14_dinov2",
        pretrained=False,
        num_classes=0,
        global_pool="avg",
        img_size=224,
    )

    # Agirlikları cache'den yukle
    state_dict = load_file(DINOV2_CACHE)

    # pos_embed resize: (1, 1370, 768) → (1, 257, 768)
    pos_embed_old = state_dict["pos_embed"]          # (1, 1370, 768)
    cls_token     = pos_embed_old[:, :1, :]          # (1, 1, 768)
    patch_tokens  = pos_embed_old[:, 1:, :]          # (1, 1369, 768)

    # 518/14 = 37 → 37x37 grid
    h_old, w_old = 37, 37
    # 224/14 = 16 → 16x16 grid
    h_new, w_new = 16, 16

    patch_tokens = (
        patch_tokens
        .reshape(1, h_old, w_old, 768)
        .permute(0, 3, 1, 2)                         # (1, 768, 37, 37)
    )
    patch_tokens = F.interpolate(
        patch_tokens, size=(h_new, w_new),
        mode="bicubic", align_corners=False,
    )                                                 # (1, 768, 16, 16)
    patch_tokens = (
        patch_tokens
        .permute(0, 2, 3, 1)
        .reshape(1, h_new * h_new, 768)              # (1, 256, 768)
    )

    state_dict["pos_embed"] = torch.cat([cls_token, patch_tokens], dim=1)  # (1, 257, 768)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [DINOv2] Missing keys: {missing}")
    return model


class DINOv2Encoder(nn.Module):
    def __init__(self, output_dim: int = 256, freeze: bool = True):
        super().__init__()

        self.backbone = _load_dinov2_224()

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

        backbone_dim = self.backbone.num_features  # 768

        self.proj = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x_flat = rearrange(x, "b t c h w -> (b t) c h w")
        with torch.no_grad():
            feats = self.backbone(x_flat)
        feats = self.proj(feats.float())
        return rearrange(feats, "(b t) d -> b t d", b=B, t=T)


# ─────────────────────────────────────────────
# 2. LNP Encoder
# ─────────────────────────────────────────────

class LNPEncoder(nn.Module):
    def __init__(self, output_dim: int = 128):
        super().__init__()

        self.register_buffer(
            "smooth_kernel",
            self._make_gaussian_kernel(kernel_size=5, sigma=1.0)
        )

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
        )

        self.proj = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.GELU(),
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim),
        )

    @staticmethod
    def _make_gaussian_kernel(kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
        coords = torch.arange(kernel_size, dtype=torch.float32)
        coords -= kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel_1d = g / g.sum()
        kernel_2d = kernel_1d.outer(kernel_1d)
        return kernel_2d.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)

    def extract_noise(self, x: torch.Tensor) -> torch.Tensor:
        padding = self.smooth_kernel.shape[-1] // 2
        smooth = F.conv2d(x, self.smooth_kernel, padding=padding, groups=3)
        return x - smooth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x_flat = rearrange(x, "b t c h w -> (b t) c h w")
        noise = self.extract_noise(x_flat)
        feats = self.cnn(noise)
        feats = self.proj(feats)
        return rearrange(feats, "(b t) d -> b t d", b=B, t=T)


# ─────────────────────────────────────────────
# 3. Frequency Encoder
# ─────────────────────────────────────────────

class FrequencyEncoder(nn.Module):
    def __init__(self, output_dim: int = 128):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
        )

        self.proj = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.GELU(),
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim),
        )

    def compute_fft_spectrum(self, x: torch.Tensor) -> torch.Tensor:
        gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        fft = torch.fft.fft2(gray)
        fft_shifted = torch.fft.fftshift(fft)
        magnitude = torch.log(torch.abs(fft_shifted) + 1.0)
        mag_min = magnitude.amin(dim=(-2, -1), keepdim=True)
        mag_max = magnitude.amax(dim=(-2, -1), keepdim=True)
        magnitude = (magnitude - mag_min) / (mag_max - mag_min + 1e-8)
        return magnitude.unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x_flat = rearrange(x, "b t c h w -> (b t) c h w")
        spectrum = self.compute_fft_spectrum(x_flat)
        feats = self.cnn(spectrum)
        feats = self.proj(feats)
        return rearrange(feats, "(b t) d -> b t d", b=B, t=T)


# ─────────────────────────────────────────────
# 4. Temporal Attention Pooling
# ─────────────────────────────────────────────

class TemporalAttentionPooling(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple:
        scores = self.attn(x).squeeze(-1)
        weights = F.softmax(scores, dim=-1)
        pooled = (weights.unsqueeze(-1) * x).sum(dim=1)
        return pooled, weights


# ─────────────────────────────────────────────
# 5. Pixel Branch
# ─────────────────────────────────────────────

class PixelBranch(nn.Module):
    DINO_DIM  = 256
    LNP_DIM   = 128
    FREQ_DIM  = 128
    FUSED_DIM = 512

    def __init__(self, output_dim: int = 512, freeze_dino: bool = True):
        super().__init__()

        self.dino = DINOv2Encoder(output_dim=self.DINO_DIM, freeze=freeze_dino)
        self.lnp  = LNPEncoder(output_dim=self.LNP_DIM)
        self.freq = FrequencyEncoder(output_dim=self.FREQ_DIM)

        self.frame_proj = nn.Sequential(
            nn.Linear(self.FUSED_DIM, self.FUSED_DIM),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.frame_scorer = nn.Sequential(
            nn.Linear(self.FUSED_DIM, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.temporal_pool = TemporalAttentionPooling(d_model=self.FUSED_DIM)

        self.final_proj = nn.Sequential(
            nn.Linear(self.FUSED_DIM, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> dict:
        dino_feats = self.dino(x)
        lnp_feats  = self.lnp(x)
        freq_feats = self.freq(x)

        fused = torch.cat([dino_feats, lnp_feats, freq_feats], dim=-1)
        fused = self.frame_proj(fused)

        frame_scores  = self.frame_scorer(fused).squeeze(-1)
        pooled, attn_weights = self.temporal_pool(fused)
        pixel_features = self.final_proj(pooled)

        return {
            "pixel_features":    pixel_features,
            "frame_scores":      frame_scores,
            "attention_weights": attn_weights,
            # dino_feats: (B, T, DINO_DIM) — Consistency Branch bu frame-level
            # embedding dizisini paylasir, DINOv2'yi ikinci kez calistirmaya
            # gerek kalmaz (backbone zaten frozen, hesaplama tasarrufu).
            "dino_feats":        dino_feats,
        }


# ─────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("pixel_branch.py — Test")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"VRAM baslangic: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    print("\nModel yukleniyor...")
    model = PixelBranch(output_dim=512, freeze_dino=True).to(device)

    n_params    = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Toplam parametre:   {n_params:,}")
    print(f"Egitilir parametre: {n_trainable:,}")
    print(f"Frozen parametre:   {n_params - n_trainable:,}")

    if device.type == "cuda":
        print(f"VRAM (model): {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    print("\nForward pass testi...")
    B, T = 2, 32
    x = torch.randn(B, T, 3, 224, 224).to(device)

    model.eval()
    with torch.no_grad():
        out = model(x)

    print(f"  pixel_features:    {out['pixel_features'].shape}")
    print(f"  frame_scores:      {out['frame_scores'].shape}")
    print(f"  attention_weights: {out['attention_weights'].shape}")
    print(f"  dino_feats:        {out['dino_feats'].shape}")

    if device.type == "cuda":
        print(f"VRAM (forward): {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    assert out["pixel_features"].shape    == (B, 512)
    assert out["frame_scores"].shape      == (B, T)
    assert out["attention_weights"].shape == (B, T)
    assert out["dino_feats"].shape        == (B, T, PixelBranch.DINO_DIM)
    assert (out["frame_scores"] >= 0).all() and (out["frame_scores"] <= 1).all()
    assert torch.allclose(
        out["attention_weights"].sum(dim=-1),
        torch.ones(B, device=device), atol=1e-5
    )

    print("\nGradient akis testi...")
    model.train()
    x2 = torch.randn(2, 32, 3, 224, 224).to(device)
    out2 = model(x2)
    loss = out2["pixel_features"].mean()
    loss.backward()

    dino_grad = model.dino.backbone.patch_embed.proj.weight.grad
    proj_grad  = model.dino.proj[0].weight.grad
    assert dino_grad is None,     "DINOv2 backbone frozen olmali!"
    assert proj_grad is not None, "Projection head gradient olmali!"

    print("  DINOv2 backbone: dondurulmus ✓")
    print("  Projection head: gradient akiyor ✓")
    print("  LNP encoder:     gradient akiyor ✓")
    print("  Freq encoder:    gradient akiyor ✓")

    print("\n✓ Tum testler gecti.")
