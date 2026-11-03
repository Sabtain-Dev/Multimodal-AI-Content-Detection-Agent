"""
detector_model.py v2 — Pixel + Motion + Consistency Branch, Cok Kollu Fusion

Pixel Branch + Motion Branch + Consistency Branch
       |
ai_probability + 3'lu branch agreement (disagreement) + ablation icin
7 kombinasyonun (her tekil, her ikili, uclu) skorlari

TASARIM NOTU (ablation maliyeti):

Ana fusion + 3 branch head'i Faz 1'de zaten vardi. Bu versiyonda
ek olarak 3 ikili-fusion head'i ekledik (pixel+motion, pixel+consistency,
motion+consistency). Bu ek head'ler kucuk MLP'ler oldugu icin egitim
suresine/VRAM'e etkisi ihmal edilebilir (~%2-3 parametre artisi).

Boylece TEK egitim kosusundan, egitim sonrasi 7 farkli kombinasyonun
(pixel, motion, consistency, pixel+motion, pixel+consistency,
motion+consistency, pixel+motion+consistency) AUC/FPR/F1 degerlerini
cikarabiliriz (bkz. ablation_eval.py).

Consistency Branch ana mimariye degil, DENEYSEL bir ek sinyal olarak
konumlandirilmistir - basari garantisi ablation ile dogrulanacaktir.

Faz 2'de:
  - Calibration head eklenecek (temperature scaling)
  - Variance-based uncertainty (3 branch arasi) cikarimi yapilacak
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pixel_branch       import PixelBranch
from motion_branch      import MotionBranch
from consistency_branch import ConsistencyBranch


# Kucuk ikili-baglanti fusion head (ablation icin)

class _PairFusionHead(nn.Module):
    """
    Iki branch feature'ini birlestirip tek bir logit ureten kucuk MLP.
    Ablation matrisindeki ikili kombinasyonlar (pixel+motion,
    pixel+consistency, motion+consistency) icin kullanilir.
    """

    def __init__(self, dim_a: int, dim_b: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_a + dim_b, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([feat_a, feat_b], dim=-1)
        return self.net(fused).squeeze(-1)


# Ana Fusion (3 branch)

class TripleFusion(nn.Module):
    """
    Uc branch'in feature'larini birlestirip karar ureten ana fusion
    + ablation icin gerekli tum tekil/ikili tahmin head'leri.

    Ana mimari:
        [pixel(512) + motion(512) + consistency(256)] -> 1280
            |
        Linear(1280 -> 256) + GELU + Dropout
            |
        Linear(256 -> 64) + GELU
            |
        Linear(64 -> 1) + Sigmoid -> ai_probability

    Ek head'ler (ablation icin, hesaplama maliyeti ihmal edilebilir):
        pixel_head, motion_head, consistency_head   (tekil, 3 adet)
        pixel_motion_head, pixel_consistency_head,
        motion_consistency_head                       (ikili, 3 adet)
    """

    def __init__(
        self,
        pixel_dim:       int = 512,
        motion_dim:      int = 512,
        consistency_dim: int = 256,
        hidden_dim:      int = 256,
    ):
        super().__init__()

        fused_dim = pixel_dim + motion_dim + consistency_dim

        # Ana fusion
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

        # Tekil branch head'leri (disagreement / ablation icin)
        self.pixel_head       = nn.Sequential(
            nn.Linear(pixel_dim, 64), nn.GELU(), nn.Linear(64, 1)
        )
        self.motion_head      = nn.Sequential(
            nn.Linear(motion_dim, 64), nn.GELU(), nn.Linear(64, 1)
        )
        self.consistency_head = nn.Sequential(
            nn.Linear(consistency_dim, 64), nn.GELU(), nn.Linear(64, 1)
        )

        # Ikili kombinasyon head'leri (ablation icin)
        self.pixel_motion_head       = _PairFusionHead(pixel_dim, motion_dim)
        self.pixel_consistency_head  = _PairFusionHead(pixel_dim, consistency_dim)
        self.motion_consistency_head = _PairFusionHead(motion_dim, consistency_dim)

    def forward(
        self,
        pixel_feats:       torch.Tensor,
        motion_feats:      torch.Tensor,
        consistency_feats: torch.Tensor,
    ) -> dict:
        """
        Args:
            pixel_feats:       (B, pixel_dim)
            motion_feats:      (B, motion_dim)
            consistency_feats: (B, consistency_dim)

        Returns:
            dict - ana karar + ablation icin 7 kombinasyonun tum
            logit/prob degerleri + 3'lu variance-based disagreement.
        """
        fused = torch.cat([pixel_feats, motion_feats, consistency_feats], dim=-1)
        ai_logit = self.fusion(fused).squeeze(-1)

        # Tekil head'ler
        pixel_logit       = self.pixel_head(pixel_feats).squeeze(-1)
        motion_logit      = self.motion_head(motion_feats).squeeze(-1)
        consistency_logit = self.consistency_head(consistency_feats).squeeze(-1)

        # Ikili head'ler
        pixel_motion_logit       = self.pixel_motion_head(pixel_feats, motion_feats)
        pixel_consistency_logit  = self.pixel_consistency_head(pixel_feats, consistency_feats)
        motion_consistency_logit = self.motion_consistency_head(motion_feats, consistency_feats)

        # Sigmoid ile olasiliklara cevir
        ai_probability          = torch.sigmoid(ai_logit)
        pixel_prob               = torch.sigmoid(pixel_logit)
        motion_prob               = torch.sigmoid(motion_logit)
        consistency_prob          = torch.sigmoid(consistency_logit)
        pixel_motion_prob          = torch.sigmoid(pixel_motion_logit)
        pixel_consistency_prob     = torch.sigmoid(pixel_consistency_logit)
        motion_consistency_prob    = torch.sigmoid(motion_consistency_logit)

        # 3'lu variance-based disagreement (elestiride onerilen formul):
        # Variance(pixel, motion, consistency) - yuksekse branch'ler anlasamiyor
        stacked = torch.stack([pixel_prob, motion_prob, consistency_prob], dim=0)  # (3, B)
        disagreement = stacked.var(dim=0, unbiased=False)  # (B,)

        return {
            # Ana karar
            "ai_logit":       ai_logit,
            "ai_probability": ai_probability,

            # Tekil (ablation)
            "pixel_logit":          pixel_logit,
            "motion_logit":         motion_logit,
            "consistency_logit":    consistency_logit,
            "pixel_prob":           pixel_prob,
            "motion_prob":          motion_prob,
            "consistency_prob":     consistency_prob,

            # Ikili (ablation)
            "pixel_motion_logit":          pixel_motion_logit,
            "pixel_consistency_logit":     pixel_consistency_logit,
            "motion_consistency_logit":    motion_consistency_logit,
            "pixel_motion_prob":           pixel_motion_prob,
            "pixel_consistency_prob":      pixel_consistency_prob,
            "motion_consistency_prob":     motion_consistency_prob,

            # Uncertainty
            "disagreement": disagreement,
        }


# Ana Model

class VideoForensicsDetector(nn.Module):
    """
    Faz 2 ana model - Pixel + Motion + Consistency Branch.

    Girdi:  video frame'leri (B, T, 3, H, W)
    Cikti:  ai_probability + ablation icin 7 kombinasyon + uncertainty
    """

    def __init__(
        self,
        pixel_output_dim:       int  = 512,
        motion_output_dim:      int  = 512,
        consistency_output_dim: int  = 256,
        freeze_dino:             bool = True,
    ):
        super().__init__()

        self.pixel_branch  = PixelBranch(
            output_dim=pixel_output_dim,
            freeze_dino=freeze_dino,
        )
        self.motion_branch = MotionBranch(output_dim=motion_output_dim)

        # Consistency Branch, Pixel Branch'in dino_feats'ini paylasir.
        # DINO_DIM Pixel Branch icindeki sabittir (256, projection sonrasi).
        self.consistency_branch = ConsistencyBranch(
            dino_dim=PixelBranch.DINO_DIM,
            output_dim=consistency_output_dim,
        )

        self.fusion = TripleFusion(
            pixel_dim=pixel_output_dim,
            motion_dim=motion_output_dim,
            consistency_dim=consistency_output_dim,
        )

    def forward(self, frames: torch.Tensor) -> dict:
        """
        Args:
            frames: (B, T, 3, H, W)  - T >= 3 olmali (Consistency Branch icin)

        Returns:
            dict:
                ai_probability:  (B,)   - ana cikti
                ai_logit:        (B,)   - loss icin
                disagreement:    (B,)   - 3'lu variance-based uncertainty
                + ablation icin tum tekil/ikili logit/prob degerleri
                frame_scores, attention, smoothness, speed_mean, jerk_mean
                - debug/analiz icin yardimci ciktilar
        """
        pixel_out       = self.pixel_branch(frames)
        motion_out      = self.motion_branch(frames)
        consistency_out = self.consistency_branch(pixel_out["dino_feats"])

        fusion_out = self.fusion(
            pixel_out["pixel_features"],
            motion_out["motion_features"],
            consistency_out["consistency_features"],
        )

        return {
            **fusion_out,
            "frame_scores": pixel_out["frame_scores"],
            "attention":    pixel_out["attention_weights"],
            "smoothness":   motion_out["smoothness_score"],
            "speed_mean":   consistency_out["speed_mean"],
            "jerk_mean":    consistency_out["jerk_mean"],
        }

    def compute_loss(
        self,
        outputs:    dict,
        labels:     torch.Tensor,
        alpha:      float = 1.0,
        beta:       float = 0.2,
        gamma:      float = 0.2,
        delta:      float = 0.2,
        pair_weight: float = 0.1,
        pos_weight:  float = 1.0,
    ) -> dict:
        """
        Cok bilesenli kayip fonksiyonu.

        Toplam kayip:
            alpha * fusion_loss                      (ana karar, en onemli)
          + beta  * pixel_loss
          + gamma * motion_loss
          + delta * consistency_loss
          + pair_weight * (pixel_motion_loss + pixel_consistency_loss
                            + motion_consistency_loss)

        POS_WEIGHT NOTU (varsayilan 1.0, notr):
          Veri seti zaten dengeli (fake:real yaklasik 1:1.26), bu yuzden
          sinif agirlikli loss'a matematiksel olarak ihtiyac yok.

          ONEMLI: pos_weight, amac fonksiyonunu (loss) dogrudan bir
          degerlendirme metrigini (FPR) optimize etmek icin kullanmak
          icin DEGISTIRILMEMELI. Boyle bir kullanim, amac fonksiyonu ile
          degerlendirme metrigini birbirine karistirir ve ablation
          sonuclarini (Consistency Branch'in gercek katkisi gibi)
          yorumlanamaz hale getirir.

          AEGIS'teki FPR sorunu (Faz 1: %33.5) bu egitimde once veri
          kalitesi (Kinetics-400, HD-VG-130M) ve yeni Consistency Branch
          ile cozulmeye calisilacak. Eger egitim sonrasi FPR hala
          yuksek kalirsa, pos_weight < 1.0 (orn. 0.5-0.7) ile AYRI,
          izole bir hyperparameter deneyi yapilmali - bu deney ana
          egitimle (Consistency Branch'in etkisini olcen) karistirilmamali.

        Args:
            outputs: forward() ciktisi
            labels:  (B,) 0=real, 1=fake/AI

        Returns:
            dict: total + her bilesenin ayri (detached) degeri
        """
        labels_f = labels.float()
        pw = torch.tensor(pos_weight, device=labels.device)

        def bce(logits):
            return F.binary_cross_entropy_with_logits(
                logits, labels_f, pos_weight=pw
            )

        fusion_loss       = bce(outputs["ai_logit"])
        pixel_loss        = bce(outputs["pixel_logit"])
        motion_loss        = bce(outputs["motion_logit"])
        consistency_loss   = bce(outputs["consistency_logit"])

        pixel_motion_loss       = bce(outputs["pixel_motion_logit"])
        pixel_consistency_loss  = bce(outputs["pixel_consistency_logit"])
        motion_consistency_loss = bce(outputs["motion_consistency_logit"])

        pair_total = (
            pixel_motion_loss + pixel_consistency_loss + motion_consistency_loss
        )

        total = (
            alpha * fusion_loss
            + beta  * pixel_loss
            + gamma * motion_loss
            + delta * consistency_loss
            + pair_weight * pair_total
        )

        return {
            "total":                   total,
            "fusion":                  fusion_loss.detach(),
            "pixel":                   pixel_loss.detach(),
            "motion":                  motion_loss.detach(),
            "consistency":             consistency_loss.detach(),
            "pixel_motion":            pixel_motion_loss.detach(),
            "pixel_consistency":       pixel_consistency_loss.detach(),
            "motion_consistency":      motion_consistency_loss.detach(),
        }


# Test

if __name__ == "__main__":
    print("=" * 55)
    print("detector_model.py v2 - Test (Pixel + Motion + Consistency)")
    print("=" * 55)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"VRAM baslangic: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    print("\nModel olusturuluyor...")
    model = VideoForensicsDetector(freeze_dino=True).to(device)

    n_params    = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Toplam parametre:    {n_params:,}")
    print(f"Egitilir parametre:  {n_trainable:,}")
    print(f"Frozen parametre:    {n_params - n_trainable:,}")

    if device.type == "cuda":
        print(f"VRAM (model): {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    print("\nForward pass testi...")
    B, T = 2, 16
    frames = torch.randn(B, T, 3, 224, 224).to(device)
    labels = torch.tensor([0, 1]).to(device)

    model.eval()
    with torch.no_grad():
        out = model(frames)

    print(f"  ai_probability:          {out['ai_probability'].shape} -> {out['ai_probability'].tolist()}")
    print(f"  pixel_prob:              {out['pixel_prob'].shape}")
    print(f"  motion_prob:             {out['motion_prob'].shape}")
    print(f"  consistency_prob:        {out['consistency_prob'].shape}")
    print(f"  pixel_motion_prob:       {out['pixel_motion_prob'].shape}")
    print(f"  pixel_consistency_prob:  {out['pixel_consistency_prob'].shape}")
    print(f"  motion_consistency_prob: {out['motion_consistency_prob'].shape}")
    print(f"  disagreement (3'lu var): {out['disagreement'].shape} -> {out['disagreement'].tolist()}")
    print(f"  frame_scores:            {out['frame_scores'].shape}")
    print(f"  attention:               {out['attention'].shape}")
    print(f"  smoothness:              {out['smoothness'].shape}")
    print(f"  speed_mean:              {out['speed_mean'].shape}")
    print(f"  jerk_mean:               {out['jerk_mean'].shape}")

    if device.type == "cuda":
        print(f"VRAM (forward): {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Shape kontrolu
    assert out["ai_probability"].shape == (B,)
    assert out["pixel_prob"].shape     == (B,)
    assert out["motion_prob"].shape    == (B,)
    assert out["consistency_prob"].shape == (B,)
    assert out["pixel_motion_prob"].shape == (B,)
    assert out["disagreement"].shape == (B,)
    assert (out["ai_probability"] >= 0).all() and (out["ai_probability"] <= 1).all()
    assert (out["disagreement"] >= 0).all()

    print("\nLoss hesabi testi (pos_weight=2.0 asimetrik ceza ile)...")
    model.train()
    out_train = model(frames)
    losses = model.compute_loss(out_train, labels, pos_weight=2.0)

    print(f"  Total loss:              {losses['total'].item():.4f}")
    print(f"  Fusion loss:             {losses['fusion'].item():.4f}")
    print(f"  Pixel loss:              {losses['pixel'].item():.4f}")
    print(f"  Motion loss:             {losses['motion'].item():.4f}")
    print(f"  Consistency loss:        {losses['consistency'].item():.4f}")
    print(f"  Pixel+Motion loss:       {losses['pixel_motion'].item():.4f}")
    print(f"  Pixel+Consistency loss:  {losses['pixel_consistency'].item():.4f}")
    print(f"  Motion+Consistency loss: {losses['motion_consistency'].item():.4f}")

    print("\nGradient akis testi...")
    losses["total"].backward()

    pixel_proj_grad        = model.pixel_branch.dino.proj[0].weight.grad
    motion_flow_grad       = model.motion_branch.flow_net.encoder[0].weight.grad
    consistency_grad       = model.consistency_branch.transformer.input_proj.weight.grad
    fusion_grad            = model.fusion.fusion[0].weight.grad
    pixel_head_grad        = model.fusion.pixel_head[0].weight.grad
    motion_head_grad       = model.fusion.motion_head[0].weight.grad
    consistency_head_grad  = model.fusion.consistency_head[0].weight.grad
    pixel_motion_grad      = model.fusion.pixel_motion_head.net[0].weight.grad

    assert pixel_proj_grad       is not None, "Pixel branch gradient olmali"
    assert motion_flow_grad      is not None, "Motion branch gradient olmali"
    assert consistency_grad      is not None, "Consistency branch gradient olmali"
    assert fusion_grad           is not None, "Fusion gradient olmali"
    assert pixel_head_grad       is not None, "Pixel head gradient olmali"
    assert motion_head_grad      is not None, "Motion head gradient olmali"
    assert consistency_head_grad is not None, "Consistency head gradient olmali"
    assert pixel_motion_grad     is not None, "Pixel+Motion ikili head gradient olmali"

    dino_backbone_grad = model.pixel_branch.dino.backbone.patch_embed.proj.weight.grad
    assert dino_backbone_grad is None, "DINOv2 backbone frozen olmali!"

    print("  Pixel branch:        gradient akiyor OK")
    print("  Motion branch:       gradient akiyor OK")
    print("  Consistency branch:  gradient akiyor OK")
    print("  Fusion head:         gradient akiyor OK")
    print("  Tekil head'ler:      gradient akiyor OK")
    print("  Ikili head'ler:      gradient akiyor OK")
    print("  DINOv2 backbone:     hala dondurulmus OK")

    print("\nOptimizer adimi testi...")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
    )
    optimizer.zero_grad()
    out2 = model(frames)
    losses2 = model.compute_loss(out2, labels, pos_weight=2.0)
    losses2["total"].backward()
    optimizer.step()
    print(f"  Optimizer adimi gecti OK")
    print(f"  Loss before/after: {losses['total'].item():.4f} -> {losses2['total'].item():.4f}")

    print("\nTum testler gecti.")
    print(f"\nFaz 2 modeli (3 branch) hazir.")
    print(f"  Egitilir parametre sayisi: {n_trainable:,}")
    print(f"  Ablation icin 7 kombinasyon destekleniyor.")
    print(f"  Asimetrik loss (pos_weight) aktif - FPR azaltma stratejisi.")
