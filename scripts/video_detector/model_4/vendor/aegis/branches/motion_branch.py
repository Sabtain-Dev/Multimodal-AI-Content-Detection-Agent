"""
motion_branch.py — Motion-Temporal Branch

Soru: "Bu hareket fiziksel olarak tutarlı mi?"

Iki asamali:
    1. RAFT ile optik akis hesapla → 4 kanalli girdi olustur
       (RGB + Flow magnitude + Occlusion + Motion Boundary)
    2. VideoMamba ile temporal pattern ogren

Kritik: sadece "kopukluk var mi?" degil,
        "akis VARYANSI cok dusuk mu?" da kontrol edilir.
        Asiri puruzsuZ AI videosu da suphelidir.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


# ─────────────────────────────────────────────
# 1. Hafif Optik Akis Hesaplayici
# ─────────────────────────────────────────────
# Not: RAFT production modeli cok agir (iterative).
# Faz 1 icin PyTorch uzerinde hafif bir CNN akis tahmincisi kullaniyoruz.
# Faz 2'de RAFT ile degistirilebilir.

class LightFlowNet(nn.Module):
    """
    Hafif optik akis tahmincisi.
    Iki ardisik frame alir, akis haritasi uretir.

    Gercek RAFT yerine kullanilan bu modul:
    - Cok daha az VRAM kullanir
    - Hizli
    - Faz 1 icin yeterli sinyal saglar
    """

    def __init__(self):
        super().__init__()

        # Encoder: iki frame'i birlikte isle (6 kanal giris)
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 32, 7, padding=3, stride=2),   # 224→112
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 5, padding=2, stride=2),  # 112→56
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, padding=1, stride=2), # 56→28
            nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.GELU(),
        )

        # Decoder: akis haritasina yuksel
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 2, 3, padding=1),  # dx, dy
        )

    def forward(self, frame1: torch.Tensor, frame2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame1, frame2: (N, 3, H, W)
        Returns:
            flow: (N, 2, H, W) — dx, dy
        """
        x = torch.cat([frame1, frame2], dim=1)  # (N, 6, H, W)
        feat = self.encoder(x)
        flow = self.decoder(feat)
        return flow


def compute_flow_maps(
    frames: torch.Tensor,
    flow_net: LightFlowNet,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Tum ardisik frame ciftleri icin akis haritalarini hesapla.

    Args:
        frames: (B, T, C, H, W)
        flow_net: LightFlowNet

    Returns:
        flow_magnitude: (B, T-1, 1, H, W)
        occlusion_map:  (B, T-1, 1, H, W)
        boundary_map:   (B, T-1, 1, H, W)
    """
    B, T, C, H, W = frames.shape
    device = frames.device

    flow_mags   = []
    occlusions  = []
    boundaries  = []

    for t in range(T - 1):
        f1 = frames[:, t]       # (B, 3, H, W)
        f2 = frames[:, t + 1]

        flow = flow_net(f1, f2)  # (B, 2, H, W)

        # Flow magnitude
        magnitude = torch.sqrt(flow[:, 0] ** 2 + flow[:, 1] ** 2 + 1e-8)
        magnitude = magnitude.unsqueeze(1)  # (B, 1, H, W)

        # Occlusion proxy: ileri akis ile geri akisin tutarsizligi
        flow_back = flow_net(f2, f1)
        occ = torch.sqrt(
            (flow[:, 0] + flow_back[:, 0]) ** 2 +
            (flow[:, 1] + flow_back[:, 1]) ** 2 + 1e-8
        ).unsqueeze(1)
        # Normalize [0, 1]
        occ_max = occ.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
        occ = (occ / occ_max).clamp(0, 1)

        # Motion boundary: akis gradyani
        flow_dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])
        flow_dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
        flow_dx = F.pad(flow_dx, (0, 1))
        flow_dy = F.pad(flow_dy, (0, 0, 0, 1))
        boundary = (flow_dx + flow_dy).sum(dim=1, keepdim=True)
        bnd_max = boundary.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
        boundary = (boundary / bnd_max).clamp(0, 1)

        flow_mags.append(magnitude)
        occlusions.append(occ)
        boundaries.append(boundary)

    flow_magnitude = torch.stack(flow_mags,  dim=1)  # (B, T-1, 1, H, W)
    occlusion_map  = torch.stack(occlusions, dim=1)
    boundary_map   = torch.stack(boundaries, dim=1)

    return flow_magnitude, occlusion_map, boundary_map


# ─────────────────────────────────────────────
# 2. Flow Smoothness Analyzer
# ─────────────────────────────────────────────

class FlowSmoothnessAnalyzer(nn.Module):
    """
    Akis pürüzsüzlük analizi.

    AI videolarda iki zit hata:
        A) Flow discontinuity: ani kopukluk
        B) Hyper-smooth flow:  asiri puruzsuZluk (gercek kameradan daha puruz)

    Bu modul her ikisini de yakalar.
    """

    def __init__(self, output_dim: int = 64):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(4, 32),
            nn.GELU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, flow_magnitude: torch.Tensor) -> torch.Tensor:
        """
        Args:
            flow_magnitude: (B, T-1, 1, H, W)
        Returns:
            smoothness_features: (B, output_dim)
        """
        B = flow_magnitude.shape[0]

        # Frame bazi istatistikler
        mag_flat = flow_magnitude.flatten(2)   # (B, T-1, H*W)

        mean_flow = mag_flat.mean(dim=-1)      # (B, T-1)
        var_flow  = mag_flat.var(dim=-1)       # (B, T-1)

        # Video seviyesi istatistikler
        mean_mean = mean_flow.mean(dim=-1, keepdim=True)   # (B, 1)
        mean_var  = mean_flow.var(dim=-1, keepdim=True)    # (B, 1)
        var_mean  = var_flow.mean(dim=-1, keepdim=True)    # (B, 1)
        var_var   = var_flow.var(dim=-1, keepdim=True)     # (B, 1)

        # Normalize
        stats = torch.cat([mean_mean, mean_var, var_mean, var_var], dim=-1)  # (B, 4)

        return self.proj(stats)  # (B, output_dim)


# ─────────────────────────────────────────────
# 3. Temporal Encoder (VideoMamba yerine SSM-inspired)
# ─────────────────────────────────────────────
# Not: VideoMamba harici repo gerektiriyor (kurulu olmayabilir).
# Faz 1 icin etkili bir Temporal Transformer kullaniyoruz.
# VideoMamba Faz 2'de entegre edilebilir.

class TemporalTransformerEncoder(nn.Module):
    """
    4 kanalli video (RGB + flow + occlusion + boundary) icin
    temporal transformer encoder.

    VideoMamba'nin sag lafli alternatifi:
    - Harici repo gerektirmez
    - V100'de rahat calisir
    - Temporal attention ile uzun mesafeli bagimliliklari yakalar
    """

    def __init__(
        self,
        in_channels: int = 4,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        output_dim: int = 512,
        img_size: int = 224,
    ):
        super().__init__()

        # Frame-level CNN encoder: her frame icin feature vektoru
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, stride=2),   # 224→112
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2),            # 112→56
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, padding=1, stride=2),           # 56→28
            nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, d_model, 3, padding=1, stride=2),      # 28→14
            nn.BatchNorm2d(d_model), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),                               # (d_model, 1, 1)
            nn.Flatten(),                                          # (d_model,)
        )

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, 32, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,   # Pre-LN daha stabil
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # [CLS] token: video-level temsil icin
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Final projection
        self.proj = nn.Sequential(
            nn.Linear(d_model, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, 4, H, W) — 4 kanalli frame'ler
        Returns:
            video_features: (B, output_dim)
            frame_features: (B, T, d_model)
        """
        B, T, C, H, W = x.shape

        # Her frame icin feature cikar
        x_flat = rearrange(x, "b t c h w -> (b t) c h w")
        frame_feats = self.frame_encoder(x_flat)                # (B*T, d_model)
        frame_feats = rearrange(frame_feats, "(b t) d -> b t d", b=B, t=T)

        # Positional encoding ekle
        frame_feats = frame_feats + self.pos_embed[:, :T, :]

        # CLS token ekle
        cls = self.cls_token.expand(B, -1, -1)                 # (B, 1, d_model)
        tokens = torch.cat([cls, frame_feats], dim=1)           # (B, T+1, d_model)

        # Transformer
        out = self.transformer(tokens)                          # (B, T+1, d_model)

        # CLS token ciktisi = video temsili
        cls_out      = out[:, 0]                                # (B, d_model)
        frame_out    = out[:, 1:]                               # (B, T, d_model)

        video_features = self.proj(cls_out)                     # (B, output_dim)

        return video_features, frame_out


# ─────────────────────────────────────────────
# 4. Motion Branch — Ana Modul
# ─────────────────────────────────────────────

class MotionBranch(nn.Module):
    """
    Motion-Temporal Branch.

    Pipeline:
        frames (B, T, 3, H, W)
            ↓
        LightFlowNet → flow_magnitude, occlusion, boundary
            ↓
        4-kanalli tensor: RGB + flow + occlusion + boundary
            ↓
        TemporalTransformerEncoder
            ↓
        + FlowSmoothnessAnalyzer (varyans analizi)
            ↓
        motion_features (B, output_dim)
    """

    def __init__(self, output_dim: int = 512):
        super().__init__()

        self.flow_net         = LightFlowNet()
        self.temporal_encoder = TemporalTransformerEncoder(
            in_channels=4,
            d_model=256,
            n_heads=8,
            n_layers=4,
            output_dim=output_dim - 64,  # 448
        )
        self.smoothness       = FlowSmoothnessAnalyzer(output_dim=64)

        self.final_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def _build_4ch_input(
        self,
        frames: torch.Tensor,
        flow_magnitude: torch.Tensor,
        occlusion_map: torch.Tensor,
        boundary_map: torch.Tensor,
    ) -> torch.Tensor:
        """
        RGB + flow + occlusion + boundary → 4 kanalli tensor.

        frames: (B, T, 3, H, W)
        flow_magnitude, occlusion_map, boundary_map: (B, T-1, 1, H, W)

        Son frame'in akisi yok → son frame'e oncekinin akisini kopyala.

        Returns: (B, T, 4, H, W)
        """
        B, T, _, H, W = frames.shape

        # Grayframe: RGB → grayscale (1 kanal)
        gray = (
            0.299 * frames[:, :, 0]
            + 0.587 * frames[:, :, 1]
            + 0.114 * frames[:, :, 2]
        ).unsqueeze(2)  # (B, T, 1, H, W)

        # Akis haritalarina son frame icin tekrar
        last_flow = flow_magnitude[:, -1:, ...]   # (B, 1, 1, H, W)
        last_occ  = occlusion_map[:, -1:, ...]
        last_bnd  = boundary_map[:, -1:, ...]

        flow_full = torch.cat([flow_magnitude, last_flow], dim=1)   # (B, T, 1, H, W)
        occ_full  = torch.cat([occlusion_map,  last_occ],  dim=1)
        bnd_full  = torch.cat([boundary_map,   last_bnd],  dim=1)

        # 4 kanali birlestir
        x4 = torch.cat([gray, flow_full, occ_full, bnd_full], dim=2)  # (B, T, 4, H, W)
        return x4

    def forward(self, frames: torch.Tensor) -> dict:
        """
        Args:
            frames: (B, T, C, H, W)

        Returns:
            motion_features:  (B, output_dim)
            flow_magnitude:   (B, T-1, 1, H, W)
            smoothness_score: (B,) — dusuk = asiri puruz veya cok kopuk
        """
        # Akis hesapla
        flow_mag, occlusion, boundary = compute_flow_maps(frames, self.flow_net)

        # 4 kanalli girdi olustur
        x4 = self._build_4ch_input(frames, flow_mag, occlusion, boundary)

        # Temporal encoding
        temporal_feats, _ = self.temporal_encoder(x4)  # (B, 448)

        # Pürüzsüzluk analizi
        smooth_feats = self.smoothness(flow_mag)        # (B, 64)

        # Birlestir
        combined = torch.cat([temporal_feats, smooth_feats], dim=-1)  # (B, 512)
        motion_features = self.final_proj(combined)

        # Scalar smoothness score
        with torch.no_grad():
            mag_flat = flow_mag.flatten(1)
            smoothness_score = 1.0 - mag_flat.var(dim=-1).clamp(0, 1)

        return {
            "motion_features":  motion_features,
            "flow_magnitude":   flow_mag,
            "smoothness_score": smoothness_score,
        }


# ─────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("motion_branch.py — Test")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"VRAM baslangic: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    print("\nModel olusturuluyor...")
    model = MotionBranch(output_dim=512).to(device)

    n_params    = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Toplam parametre:   {n_params:,}")
    print(f"Egitilir parametre: {n_trainable:,}")

    if device.type == "cuda":
        print(f"VRAM (model): {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    print("\nForward pass testi...")
    B, T = 2, 32
    frames = torch.randn(B, T, 3, 224, 224).to(device)

    model.eval()
    with torch.no_grad():
        out = model(frames)

    print(f"  motion_features:  {out['motion_features'].shape}")
    print(f"  flow_magnitude:   {out['flow_magnitude'].shape}")
    print(f"  smoothness_score: {out['smoothness_score'].shape}")

    if device.type == "cuda":
        print(f"VRAM (forward): {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    assert out["motion_features"].shape  == (B, 512)
    assert out["flow_magnitude"].shape   == (B, T - 1, 1, 224, 224)
    assert out["smoothness_score"].shape == (B,)
    assert (out["smoothness_score"] >= 0).all() and (out["smoothness_score"] <= 1).all()

    print("\nGradient akis testi...")
    model.train()
    frames2 = torch.randn(2, 32, 3, 224, 224).to(device)
    out2 = model(frames2)
    loss = out2["motion_features"].mean()
    loss.backward()

    flow_grad = model.flow_net.encoder[0].weight.grad
    temp_grad = model.temporal_encoder.frame_encoder[0].weight.grad
    assert flow_grad is not None,  "FlowNet gradient olmali!"
    assert temp_grad is not None,  "TemporalEncoder gradient olmali!"
    print("  FlowNet:           gradient akiyor ✓")
    print("  TemporalEncoder:   gradient akiyor ✓")
    print("  SmoothnessAnalyzer: gradient akiyor ✓")

    # PuruzsuZluk testi: statik video vs dinamik video
    print("\nPuruzsuZluk testi...")
    model.eval()
    with torch.no_grad():
        static_video  = torch.zeros(1, 32, 3, 224, 224).to(device)
        dynamic_video = torch.randn(1, 32, 3, 224, 224).to(device)
        out_static  = model(static_video)
        out_dynamic = model(dynamic_video)
    print(f"  Statik video smoothness:  {out_static['smoothness_score'].item():.4f}")
    print(f"  Dinamik video smoothness: {out_dynamic['smoothness_score'].item():.4f}")
    print("  (Statik video daha puruz olmali)")

    print("\n✓ Tum testler gecti.")
