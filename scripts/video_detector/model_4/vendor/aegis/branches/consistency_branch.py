"""
consistency_branch.py — Semantic Stability Branch (Consistency Branch)

NOT: Bu branch'in basari garantisi yoktur. Pixel ve Motion Branch'lerin
tersine, literaturde dogrudan karsiligi olmayan deneysel bir hipotezi
test eder: "AI uretimi videolarda, DINOv2 embedding'lerinin zaman
icindeki davranisi (pozisyon + hiz + ivme) gercek videolardan
ayirt edilebilir sekilde farklidir."

Bu yuzden mimaride ANA katki olarak degil, ablation ile dogrulanacak
EK bir sinyal kaynagi olarak konumlandirilmistir (bkz. egitim sonrasi
ablation_eval.py).

─────────────────────────────────────────────────────────────────
TASARIM GECMISI (3 tur elestiri sonrasi nihai karar):

  v1 (ilk taslak):
    16x16 pairwise distance matrix elle hesaplanip Transformer'a verildi.
    PROBLEM: Bilgi kaybi - distance(a,b) tek bir sayi, oysa Transformer'in
    elinde a ve b ayri ayri olursa kendi attention'i ile cok daha zengin
    iliskiyi ogrenebilir.

  v2 (ham embedding serisi):
    pairwise matrix yerine [16, 768] embedding dizisi direkt Transformer'a
    verildi. Transformer kendi self-attention'i ile frame-ciftleri
    arasindaki iliskiyi ogreniyor.
    KAZANC: Bilgi kaybi yok, Transformer'in dogal gucu kullaniliyor.

  v3 (nihai - bu dosya):
    Risk: Kamera hareketi (pan/dolly) ile AI sahteligi birbirine
    karisabilir - gercek videoda da frame[0] ile frame[-1] arasi
    mesafe yuksek olabilir (Kinetics-400'de bu çok yaygin).
    Cozum: Sadece embedding pozisyonu degil, embedding'in HIZINI
    (velocity) ve IVMESINI (acceleration) de ayri kanallar olarak ver.
    Ayrica vektorlerin YONUNU (raw velocity/acceleration) ile
    BUYUKLUGUNU (speed=||velocity||, jerk=||acceleration||) ayri ayri
    besle - boylece model "ani sicrama var mi" sorusunu vektor
    yorumlamaya calismadan, dogrudan skaler bir kanaldan okuyabilir.

    Hipotez: Gercek kamera hareketinde mesafe DUZGUN/MONOTONIK artar
    (speed sabit, jerk dusuk). AI videolarinda semantik kayma genelde
    ANI SICRAMALAR seklinde olur (jerk yuksek, belirli frame'lerde
    spike). Bu fark, ham mesafe buyuklugunden daha guvenilir bir
    AI-sahtelik sinyali olmalidir (kamera hareketinden ayristirilabilir).
─────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 1. Temporal Transformer Encoder (paylasilan parca)
# ─────────────────────────────────────────────

class _TemporalTransformerEncoder(nn.Module):
    """
    Embedding + turev kanallarinin birlestirilmis dizisini isleyen
    standart bir Transformer encoder. CLS token ile sekans seviyesinde
    tek bir ozet vektor uretir.
    """

    def __init__(
        self,
        input_dim:   int,
        d_model:     int = 256,
        n_heads:     int = 4,
        n_layers:    int = 2,
        ff_dim:      int = 512,
        max_seq_len: int = 64,
        dropout:     float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.cls_token  = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed  = nn.Parameter(torch.randn(1, max_seq_len + 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm    = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, input_dim) — L = degisken sekans uzunlugu
        Returns:
            (B, d_model) — CLS token cikisi
        """
        B, L, _ = x.shape

        x = self.input_proj(x)                                   # (B, L, d_model)
        cls = self.cls_token.expand(B, -1, -1)                    # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)                            # (B, L+1, d_model)
        x = x + self.pos_embed[:, : L + 1, :]

        x = self.encoder(x)                                       # (B, L+1, d_model)
        cls_out = x[:, 0, :]                                      # (B, d_model)
        return self.norm(cls_out)


# ─────────────────────────────────────────────
# 2. Turev ozellikleri hesaplama
# ─────────────────────────────────────────────

def _compute_derivative_features(embeddings: torch.Tensor) -> dict:
    """
    DINOv2 frame embedding dizisinden velocity, acceleration,
    speed (||velocity||) ve jerk (||acceleration||) cikarir.

    Args:
        embeddings: (B, T, D) — T=16 frame, D=DINO_DIM (256, projection sonrasi)

    Returns:
        dict:
            velocity:     (B, T-1, D)
            acceleration: (B, T-2, D)
            speed:        (B, T-1, 1)  — ||velocity|| skaler, kanal olarak
            jerk:         (B, T-2, 1)  — ||acceleration|| skaler, kanal olarak
    """
    velocity     = embeddings[:, 1:, :] - embeddings[:, :-1, :]      # (B, T-1, D)
    acceleration = velocity[:, 1:, :]   - velocity[:, :-1, :]        # (B, T-2, D)

    speed = torch.norm(velocity, dim=-1, keepdim=True)               # (B, T-1, 1)
    jerk  = torch.norm(acceleration, dim=-1, keepdim=True)           # (B, T-2, 1)

    return {
        "velocity":     velocity,
        "acceleration": acceleration,
        "speed":        speed,
        "jerk":         jerk,
    }


def _build_combined_sequence(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Embedding + velocity + acceleration + speed + jerk kanallarini
    tek bir sekans olarak Transformer'a hazirlar.

    Strateji: Her "zaman adimi" turunde, o zaman adimina ait tum
    bilgileri (pozisyon, hiz, ivme, hiz-buyuklugu, ivme-buyuklugu)
    CONCAT ederek tek bir token haline getiriyoruz. Boylece sekans
    uzunlugu T-2 (en kisa turev dizisine gore hizalanmis) olur ve
    her token zengin, o anin tam "hareket durumunu" tasir.

    Args:
        embeddings: (B, T, D)

    Returns:
        combined: (B, T-2, D + D + D + 1 + 1) = (B, T-2, 3D+2)
    """
    derivs = _compute_derivative_features(embeddings)

    T = embeddings.shape[1]
    L = T - 2  # en kisa dizi (acceleration) ile hizala

    pos_aligned   = embeddings[:, 2:, :]               # (B, L, D)   - t=2..T-1
    vel_aligned   = derivs["velocity"][:, 1:, :]        # (B, L, D)   - t=2..T-1
    acc_aligned   = derivs["acceleration"]              # (B, L, D)   - t=2..T-1
    speed_aligned = derivs["speed"][:, 1:, :]           # (B, L, 1)
    jerk_aligned  = derivs["jerk"]                      # (B, L, 1)

    combined = torch.cat(
        [pos_aligned, vel_aligned, acc_aligned, speed_aligned, jerk_aligned],
        dim=-1,
    )  # (B, L, 3D+2)

    return combined


# ─────────────────────────────────────────────
# 3. Consistency / Semantic Stability Branch
# ─────────────────────────────────────────────

class ConsistencyBranch(nn.Module):
    """
    Semantic Stability Branch.

    Pixel Branch'in zaten hesapladigi frame-level DINOv2 embedding
    dizisini (dino_feats, (B, T, DINO_DIM)) girdi olarak alir —
    DINOv2'yi ikinci kez calistirmaz, sifir ek backbone maliyeti.

    Pozisyon + hiz + ivme + hiz-buyuklugu + ivme-buyuklugu kanallarini
    birlestirip bir Temporal Transformer'a verir, CLS token cikisini
    "consistency_features" olarak dondurur.

    Pixel Branch'ten farki:  tek kareye degil, T boyunca DAVRANISA bakar.
    Motion Branch'ten farki: piksel-seviye optik akisa degil, YUKSEK
                              SEVIYE SEMANTIK embedding'in zaman icindeki
                              davranisina bakar (uzun-mesafeli iliski).
    """

    def __init__(
        self,
        dino_dim:    int = 256,
        output_dim:  int = 256,
        d_model:     int = 256,
        n_heads:     int = 4,
        n_layers:    int = 2,
        max_frames:  int = 64,
        dropout:     float = 0.1,
    ):
        super().__init__()

        self.dino_dim = dino_dim

        # combined token boyutu: pos(D) + vel(D) + acc(D) + speed(1) + jerk(1)
        combined_dim = dino_dim * 3 + 2

        self.transformer = _TemporalTransformerEncoder(
            input_dim=combined_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_seq_len=max_frames,
            dropout=dropout,
        )

        self.final_proj = nn.Sequential(
            nn.Linear(d_model, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, dino_feats: torch.Tensor) -> dict:
        """
        Args:
            dino_feats: (B, T, dino_dim) — Pixel Branch'ten paylasilan,
                        pooling'den ONCEKI frame-level DINOv2 embedding'leri.
                        T >= 3 olmali (acceleration hesabi icin en az 3 frame).

        Returns:
            dict:
                consistency_features: (B, output_dim)
                speed_mean:            (B,) — debug/analiz icin ortalama hiz
                jerk_mean:              (B,) — debug/analiz icin ortalama ivme-buyuklugu
        """
        B, T, D = dino_feats.shape
        if T < 3:
            raise ValueError(
                f"ConsistencyBranch en az 3 frame gerektirir (acceleration icin), "
                f"alinan T={T}"
            )

        combined = _build_combined_sequence(dino_feats)   # (B, T-2, 3D+2)
        pooled   = self.transformer(combined)             # (B, d_model)
        consistency_features = self.final_proj(pooled)    # (B, output_dim)

        derivs = _compute_derivative_features(dino_feats)
        speed_mean = derivs["speed"].mean(dim=(1, 2))      # (B,)
        jerk_mean  = derivs["jerk"].mean(dim=(1, 2))        # (B,)

        return {
            "consistency_features": consistency_features,
            "speed_mean":            speed_mean,
            "jerk_mean":             jerk_mean,
        }


# ─────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("consistency_branch.py — Test")
    print("=" * 55)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nModel olusturuluyor...")
    model = ConsistencyBranch(dino_dim=256, output_dim=256).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Toplam parametre: {n_params:,}")

    print("\nForward pass testi (gercekci T=16, DINO_DIM=256)...")
    B, T, D = 2, 16, 256
    dino_feats = torch.randn(B, T, D).to(device)

    model.eval()
    with torch.no_grad():
        out = model(dino_feats)

    print(f"  consistency_features: {out['consistency_features'].shape}")
    print(f"  speed_mean:            {out['speed_mean'].shape} → {out['speed_mean'].tolist()}")
    print(f"  jerk_mean:              {out['jerk_mean'].shape} → {out['jerk_mean'].tolist()}")

    assert out["consistency_features"].shape == (B, 256)
    assert out["speed_mean"].shape            == (B,)
    assert out["jerk_mean"].shape              == (B,)

    print("\nMinimum frame sayisi testi (T=3)...")
    dino_feats_min = torch.randn(B, 3, D).to(device)
    with torch.no_grad():
        out_min = model(dino_feats_min)
    print(f"  T=3 ile calisti ✓, consistency_features: {out_min['consistency_features'].shape}")

    print("\nHata durumu testi (T=2, yetersiz)...")
    try:
        dino_feats_bad = torch.randn(B, 2, D).to(device)
        model(dino_feats_bad)
        print("  HATA: ValueError firlatilmadi!")
    except ValueError as e:
        print(f"  Beklenen hata yakalandi ✓: {e}")

    print("\nKamera hareketi simulasyon testi...")
    print("  (duzgun/monotonik kayma vs ani sicrama - speed/jerk farki)")

    smooth_drift = torch.zeros(1, 16, D)
    base = torch.randn(D)
    for t in range(16):
        smooth_drift[0, t] = base + t * 0.1 * torch.randn(D) * 0.01 + t * 0.05
    smooth_drift = smooth_drift.to(device)

    spiky_drift = torch.zeros(1, 16, D)
    base2 = torch.randn(D)
    for t in range(16):
        spiky_drift[0, t] = base2 + (torch.randn(D) * 2.0 if t in [5, 11] else torch.zeros(D))
    spiky_drift = spiky_drift.to(device)

    with torch.no_grad():
        out_smooth = model(smooth_drift)
        out_spiky  = model(spiky_drift)

    print(f"  Duzgun hareket  - jerk_mean: {out_smooth['jerk_mean'].item():.4f}")
    print(f"  Ani sicrama     - jerk_mean: {out_spiky['jerk_mean'].item():.4f}")
    print(f"  (Ani sicramali olanin jerk degeri daha yuksek olmasi beklenir)")

    print("\nGradient akis testi...")
    model.train()
    dino_feats2 = torch.randn(2, 16, D).to(device)
    out2 = model(dino_feats2)
    loss = out2["consistency_features"].mean()
    loss.backward()

    transformer_grad = model.transformer.input_proj.weight.grad
    final_proj_grad  = model.final_proj[0].weight.grad
    assert transformer_grad is not None, "Transformer gradient olmali"
    assert final_proj_grad  is not None, "Final proj gradient olmali"
    print("  Transformer: gradient akiyor ✓")
    print("  Final proj:  gradient akiyor ✓")

    print("\n✓ Tum testler gecti.")
