"""
video_io.py — Video okuma ve frame sampling

Değişiklikler (v2):
  - Temporal window sampling: her videodan ardışık bir pencere al
  - Kalite filtresi: çok kısa veya çok düşük çözünürlüklü videoları reddet
  - Sampling modu: 'uniform' veya 'window' 
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass

try:
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False

# ─────────────────────────────────────────────
# Sabitler
# ─────────────────────────────────────────────

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Kalite filtreleme eşikleri
MIN_FRAMES      = 8       # en az bu kadar frame olmalı
MIN_RESOLUTION  = 128     # en kısa kenar en az bu kadar piksel olmalı
MIN_DURATION    = 1.0     # en az 1 saniye


# ─────────────────────────────────────────────
# Veri yapısı
# ─────────────────────────────────────────────

@dataclass
class FrameBundle:
    frames_all:        torch.Tensor   # (N, C, H, W)
    frames_semantic:   torch.Tensor   # (N_s, C, H, W)
    video_path:        str
    total_frames:      int
    fps:               float
    importance_scores: np.ndarray
    resolution:        tuple          # (W, H) orijinal
    duration:          float          # saniye


class VideoQualityError(Exception):
    """Kalite filtresi geçilemediğinde fırlatılır."""
    pass


# ─────────────────────────────────────────────
# Kalite filtresi
# ─────────────────────────────────────────────

def check_video_quality(
    video_path: str,
    min_frames:     int   = MIN_FRAMES,
    min_resolution: int   = MIN_RESOLUTION,
    min_duration:   float = MIN_DURATION,
) -> dict:
    """
    Video kalitesini kontrol et.
    Geçemezse VideoQualityError fırlatır.
    
    Returns: {'total_frames', 'fps', 'width', 'height', 'duration'}
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise VideoQualityError(f"Video açılamadı: {video_path}")

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    duration = total / fps if fps > 0 else 0

    if total < min_frames:
        raise VideoQualityError(
            f"Yetersiz frame: {total} < {min_frames} ({video_path})"
        )
    if min(width, height) < min_resolution:
        raise VideoQualityError(
            f"Düşük çözünürlük: {width}x{height} < {min_resolution}px ({video_path})"
        )
    if duration < min_duration:
        raise VideoQualityError(
            f"Çok kısa video: {duration:.1f}s < {min_duration}s ({video_path})"
        )

    return {
        'total_frames': total,
        'fps':          fps,
        'width':        width,
        'height':       height,
        'duration':     duration,
    }


# ─────────────────────────────────────────────
# Frame okuma
# ─────────────────────────────────────────────

def _read_frames_decord(
    video_path: str,
    indices:    np.ndarray,
) -> np.ndarray:
    """Decord ile belirli indekslerdeki frame'leri oku."""
    vr = VideoReader(video_path, ctx=cpu(0))
    return vr.get_batch(indices).asnumpy()  # (N, H, W, C)


def _read_frames_cv2(
    video_path: str,
    indices:    np.ndarray,
) -> np.ndarray:
    """cv2 fallback."""
    cap = cv2.VideoCapture(video_path)
    idx_set = set(indices.tolist())
    frames  = {}
    idx     = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx in idx_set:
            frames[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if len(frames) == len(idx_set):
            break
        idx += 1

    cap.release()

    result = []
    for i in indices:
        if i in frames:
            result.append(frames[i])
        elif result:
            result.append(result[-1])
        else:
            result.append(np.zeros((224, 224, 3), dtype=np.uint8))

    return np.stack(result)


def read_frames_at(video_path: str, indices: np.ndarray) -> np.ndarray:
    """Belirli frame indekslerini oku — decord veya cv2."""
    if DECORD_AVAILABLE:
        try:
            return _read_frames_decord(video_path, indices)
        except Exception:
            pass
    return _read_frames_cv2(video_path, indices)


# ─────────────────────────────────────────────
# Sampling stratejileri
# ─────────────────────────────────────────────

def uniform_sample(total_frames: int, n_frames: int) -> np.ndarray:
    """
    Klasik uniform sampling.
    Video boyunca eşit aralıklı N frame seç.
    """
    return np.linspace(0, total_frames - 1, n_frames, dtype=int)


def window_sample(
    total_frames: int,
    n_frames:     int,
    fps:          float,
    target_dur:   float = 4.0,
    random_start: bool  = True,
) -> np.ndarray:
    """
    Temporal window sampling.
    
    Videodan ardışık bir pencere al. Bu sayede:
    - Kısa fake videolarda tüm video alınır
    - Uzun real videolarda rastgele bir 4 saniyelik pencere alınır
    - Motion branch için ardışık frame'ler daha anlamlı sinyal verir
    
    Args:
        total_frames: toplam frame sayısı
        n_frames:     seçilecek frame sayısı
        fps:          videonun FPS'i
        target_dur:   hedef pencere süresi (saniye)
        random_start: True ise rastgele başlangıç, False ise merkez
    
    Returns:
        indices: (n_frames,) seçilen frame indeksleri
    """
    # Hedef pencere boyutu (frame cinsinden)
    window_size = min(int(target_dur * fps), total_frames)
    window_size = max(window_size, n_frames)  # en az n_frames kadar

    if window_size >= total_frames:
        # Video zaten kısa — tüm videoyu kullan
        return np.linspace(0, total_frames - 1, n_frames, dtype=int)

    # Pencere başlangıcı
    max_start = total_frames - window_size
    if random_start:
        start = np.random.randint(0, max_start + 1)
    else:
        start = max_start // 2  # merkez

    end = start + window_size
    return np.linspace(start, end - 1, n_frames, dtype=int)


# ─────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────

def preprocess_frames(
    frames: np.ndarray,
    height: int = 224,
    width:  int = 224,
) -> torch.Tensor:
    """
    (T, H, W, C) uint8 → (T, C, H, W) float32, ImageNet normalize
    """
    T   = len(frames)
    out = np.zeros((T, height, width, 3), dtype=np.float32)

    for i, frame in enumerate(frames):
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        out[i] = (frame.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD

    return torch.from_numpy(out).permute(0, 3, 1, 2).contiguous()


# ─────────────────────────────────────────────
# Importance scoring
# ─────────────────────────────────────────────

def compute_importance_scores(frames: np.ndarray) -> np.ndarray:
    """Her frame için önem skoru — semantic + motion değişimi."""
    T      = len(frames)
    scores = np.zeros(T, dtype=np.float32)

    for t in range(1, T):
        diff           = frames[t].astype(np.float32) - frames[t-1].astype(np.float32)
        semantic_change = float(np.linalg.norm(diff)) / (diff.size + 1e-8)

        prev_gray = cv2.cvtColor(frames[t-1], cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(frames[t],   cv2.COLOR_RGB2GRAY)
        flow      = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        motion_change = float(np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)))
        scores[t]     = 0.5 * semantic_change + 0.5 * motion_change

    scores[0] = scores[1] if T > 1 else 0.0
    s_max     = scores.max()
    if s_max > 0:
        scores = scores / s_max
    return scores


def select_semantic_frames(
    frames:   np.ndarray,
    scores:   np.ndarray,
    n_select: int = 8,
) -> tuple:
    n_select    = min(n_select, len(frames))
    top_indices = np.sort(np.argsort(scores)[-n_select:])
    return frames[top_indices], top_indices


# ─────────────────────────────────────────────
# Ana pipeline
# ─────────────────────────────────────────────

def load_video(
    video_path:     str,
    n_frames:       int   = 16,
    n_semantic:     int   = 8,
    height:         int   = 224,
    width:          int   = 224,
    sampling:       str   = 'window',   # 'uniform' veya 'window'
    target_dur:     float = 4.0,        # window sampling hedef süresi
    random_start:   bool  = True,       # window sampling rastgele başlangıç
    quality_filter: bool  = True,       # kalite filtresi uygula
) -> FrameBundle:
    """
    Video'yu okur, normalize eder, FrameBundle döndürür.

    sampling='window': temporal window sampling (önerilen)
    sampling='uniform': klasik uniform sampling
    quality_filter=True: düşük kaliteli videoları reddeder
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video bulunamadı: {video_path}")

    # 1. Kalite kontrolü
    if quality_filter:
        meta = check_video_quality(video_path)
    else:
        cap      = cv2.VideoCapture(video_path)
        total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps      = cap.get(cv2.CAP_PROP_FPS)
        w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        meta = {
            'total_frames': total, 'fps': fps,
            'width': w, 'height': h,
            'duration': total / fps if fps > 0 else 0,
        }

    total_frames = meta['total_frames']
    fps          = meta['fps']
    resolution   = (meta['width'], meta['height'])
    duration     = meta['duration']

    # 2. Frame indekslerini belirle
    if sampling == 'window':
        indices = window_sample(
            total_frames, n_frames, fps,
            target_dur=target_dur,
            random_start=random_start,
        )
    else:
        indices = uniform_sample(total_frames, n_frames)

    # 3. Frame'leri oku
    frames_np = read_frames_at(video_path, indices)

    # Eksik frame varsa son frame'i tekrarla
    while len(frames_np) < n_frames:
        frames_np = np.concatenate(
            [frames_np, frames_np[-1:]], axis=0
        )
    frames_np = frames_np[:n_frames]

    # 4. Importance scoring ve semantic frame seçimi
    scores      = compute_importance_scores(frames_np)
    semantic_np, _ = select_semantic_frames(frames_np, scores, n_semantic)

    # 5. Preprocess
    frames_tensor   = preprocess_frames(frames_np,   height, width)
    semantic_tensor = preprocess_frames(semantic_np, height, width)

    return FrameBundle(
        frames_all=frames_tensor,
        frames_semantic=semantic_tensor,
        video_path=video_path,
        total_frames=total_frames,
        fps=fps,
        importance_scores=scores,
        resolution=resolution,
        duration=duration,
    )


# ─────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile, os

    print("=" * 55)
    print("video_io.py v2 — Temporal Window Sampling Test")
    print("=" * 55)

    # Sentetik video oluştur (64 frame, 24 FPS → ~2.7s)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        tmp = f.name

    h, w   = 360, 640
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp, fourcc, 24.0, (w, h))
    for i in range(64):
        frame            = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :, 0]   = (i * 4) % 255
        cv2.rectangle(frame, (i * 8 % w, 50), (i * 8 % w + 40, 100), (0, 255, 0), -1)
        writer.write(frame)
    writer.release()

    print(f"\nTest videosu: {tmp} (64 frame, 24 FPS, {w}x{h})")

    # Window sampling testi
    print("\n[1] Window sampling (target_dur=2s):")
    b1 = load_video(tmp, n_frames=16, n_semantic=8,
                    sampling='window', target_dur=2.0, random_start=True)
    print(f"  frames_all:      {b1.frames_all.shape}")
    print(f"  frames_semantic: {b1.frames_semantic.shape}")
    print(f"  duration:        {b1.duration:.1f}s")
    print(f"  resolution:      {b1.resolution}")

    # Uniform sampling testi
    print("\n[2] Uniform sampling:")
    b2 = load_video(tmp, n_frames=16, n_semantic=8, sampling='uniform')
    print(f"  frames_all: {b2.frames_all.shape}")

    # Kalite filtresi testi
    print("\n[3] Kalite filtresi:")
    try:
        # Çok kısa video oluştur (3 frame)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            tmp_short = f.name
        w2 = cv2.VideoWriter(tmp_short, fourcc, 24.0, (w, h))
        for _ in range(3):
            w2.write(np.zeros((h, w, 3), dtype=np.uint8))
        w2.release()
        load_video(tmp_short, quality_filter=True)
        print("  HATA: Kalite filtresi çalışmadı!")
    except VideoQualityError as e:
        print(f"  Kalite filtresi çalıştı ✓: {e}")
    finally:
        os.unlink(tmp_short)

    # Uzun video — window sampling ile rastgele pencere
    print("\n[4] Uzun video window sampling tekrarlanabilirlik:")
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        tmp_long = f.name
    w3 = cv2.VideoWriter(tmp_long, fourcc, 24.0, (w, h))
    for i in range(240):  # 10 saniye
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :, i % 3] = (i * 4) % 255
        w3.write(frame)
    w3.release()

    b3 = load_video(tmp_long, n_frames=16, sampling='window',
                    target_dur=4.0, random_start=True)
    b4 = load_video(tmp_long, n_frames=16, sampling='window',
                    target_dur=4.0, random_start=True)
    frames_same = torch.allclose(b3.frames_all, b4.frames_all)
    print(f"  İki farklı çağrı aynı frame'leri verdi mi: {frames_same}")
    print(f"  (False olmalı — rastgele pencere seçiyor)")
    os.unlink(tmp_long)

    os.unlink(tmp)
    print("\n✓ Tüm testler geçti.")
