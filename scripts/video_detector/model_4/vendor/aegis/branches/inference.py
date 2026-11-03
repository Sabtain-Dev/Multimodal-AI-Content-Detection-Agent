import os
# Kendi sunucunda calistirmak icin: export AEGIS_BASE_DIR=/senin/yolun
BASE_DIR = os.environ.get("AEGIS_BASE_DIR", "/kullanici_yedek/musap.yildiz")

"""
inference.py - Tek video analizi

Kullanim:
    python3 inference.py --video /path/to/video.mp4
    python3 inference.py --video /path/to/video.mp4 --checkpoint /path/to/checkpoint.pt
    python3 inference.py --dir /path/to/video_folder/
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path

import torch
import numpy as np

THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.join(THIS_DIR, '..', 'utils')
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, UTILS_DIR)

from detector_model import VideoForensicsDetector
from video_io import load_video


# ─────────────────────────────────────────────
# Sonuc yorumlama
# ─────────────────────────────────────────────

def interpret_score(prob: float, disagreement: float) -> dict:
    """
    Skoru insan tarafindan anlasilir etikete cevir.
    """
    if disagreement > 0.4:
        confidence = "dusuk"
        note = "Branch'ler anlasamiyor — video islenmis veya bozulmus olabilir."
    elif prob > 0.85:
        confidence = "yuksek"
        note = ""
    elif prob > 0.65:
        confidence = "orta"
        note = ""
    elif prob > 0.35:
        confidence = "dusuk"
        note = "Belirsiz vaka."
    else:
        confidence = "yuksek"
        note = ""

    if prob > 0.5:
        verdict = "AI URETIMI"
        emoji   = "🤖"
    else:
        verdict = "GERCEK"
        emoji   = "✅"

    return {
        "verdict":    verdict,
        "emoji":      emoji,
        "confidence": confidence,
        "note":       note,
    }


def format_results(video_path: str, outputs: dict, elapsed: float) -> str:
    """Sonuclari okunabilir formatta yazdir."""
    prob        = outputs["ai_probability"].item()
    pixel_prob  = outputs["pixel_prob"].item()
    motion_prob = outputs["motion_prob"].item()
    disagreement = outputs["disagreement"].item()
    smoothness  = outputs["smoothness"].item()

    interp = interpret_score(prob, disagreement)

    lines = [
        f"\n{'='*55}",
        f"VIDEO: {Path(video_path).name}",
        f"{'='*55}",
        f"{interp['emoji']}  {interp['verdict']}",
        f"    Genel skor:    {prob:.3f}  ({prob*100:.1f}%)",
        f"    Guven:         {interp['confidence']}",
        f"",
        f"Branch Detaylari:",
        f"    Pixel branch:  {pixel_prob:.3f}  (texture/frekans artifact)",
        f"    Motion branch: {motion_prob:.3f}  (hareket tutarsizligi)",
        f"    Anlaşmazlik:   {disagreement:.3f}  (0=hemfikir, 1=zit)",
        f"    Puruzsuзluk:   {smoothness:.3f}  (1=cok puruzsu → suphe)",
        f"",
        f"Islem suresi: {elapsed:.2f}s",
    ]

    if interp["note"]:
        lines.append(f"⚠️  Not: {interp['note']}")

    lines.append(f"{'='*55}")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# Ana inference fonksiyonu
# ─────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> VideoForensicsDetector:
    """Model yukle."""
    model = VideoForensicsDetector(freeze_dino=True).to(device)

    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Checkpoint yukleniyor: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        epoch   = state.get("epoch", "?")
        metrics = state.get("metrics", {})
        print(f"  Epoch: {epoch} | Val AUC: {metrics.get('auc', 0):.4f}")
    else:
        print("Checkpoint bulunamadi — rastgele agirliklar kullaniliyor.")
        print("(Egitim tamamlaninca checkpoint verin)")

    model.eval()
    return model


@torch.no_grad()
def analyze_video(
    video_path: str,
    model: VideoForensicsDetector,
    device: torch.device,
) -> dict:
    """Tek video analiz et."""
    t0 = time.time()

    # Video yukle
    bundle = load_video(video_path, n_frames=16, n_semantic=8)
    frames = bundle.frames_all.unsqueeze(0).to(device)  # (1, 16, 3, 224, 224)

    # Model calistir
    outputs = model(frames)

    elapsed = time.time() - t0

    # Sonuclari CPU'ya al
    result = {
        "video_path":    video_path,
        "ai_probability":  outputs["ai_probability"].item(),
        "pixel_prob":      outputs["pixel_prob"].item(),
        "motion_prob":     outputs["motion_prob"].item(),
        "disagreement":    outputs["disagreement"].item(),
        "smoothness":      outputs["smoothness"].item(),
        "frame_scores":    outputs["frame_scores"].squeeze(0).cpu().numpy().tolist(),
        "elapsed_sec":     elapsed,
    }

    return result, outputs


def analyze_directory(
    dir_path: str,
    model: VideoForensicsDetector,
    device: torch.device,
    extensions: tuple = (".mp4", ".avi", ".mov", ".webm"),
) -> list:
    """Bir klasordeki tum videolari analiz et."""
    dir_path = Path(dir_path)
    videos = [f for f in dir_path.rglob("*") if f.suffix.lower() in extensions]

    if not videos:
        print(f"Video bulunamadi: {dir_path}")
        return []

    print(f"{len(videos)} video bulundu.\n")
    results = []

    for i, vp in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {vp.name}")
        try:
            result, outputs = analyze_video(str(vp), model, device)
            results.append(result)
            print(format_results(str(vp), outputs, result["elapsed_sec"]))
        except Exception as e:
            print(f"  HATA: {e}")

    return results


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Forensics Inference")

    parser.add_argument("--video",      type=str, default=None,
                        help="Tek video dosyasi")
    parser.add_argument("--dir",        type=str, default=None,
                        help="Video klasoru (hepsini analiz et)")
    parser.add_argument("--checkpoint", type=str,
                        default=f"{BASE_DIR}/checkpoints/phase1/checkpoint_best.pt",
                        help="Model checkpoint yolu")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Sonuclari JSON'a kaydet")
    parser.add_argument("--device",    type=str, default="auto",
                        help="cuda / cpu / auto")

    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Model
    model = load_model(args.checkpoint, device)

    # Analiz
    if args.video:
        if not Path(args.video).exists():
            print(f"HATA: Video bulunamadi: {args.video}")
            sys.exit(1)

        result, outputs = analyze_video(args.video, model, device)
        print(format_results(args.video, outputs, result["elapsed_sec"]))

        if args.output_json:
            with open(args.output_json, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\nSonuc kaydedildi: {args.output_json}")

    elif args.dir:
        results = analyze_directory(args.dir, model, device)

        if results:
            ai_count   = sum(1 for r in results if r["ai_probability"] > 0.5)
            real_count = len(results) - ai_count
            print(f"\n{'='*55}")
            print(f"OZET: {len(results)} video")
            print(f"  AI uretimi: {ai_count} ({ai_count/len(results)*100:.1f}%)")
            print(f"  Gercek:     {real_count} ({real_count/len(results)*100:.1f}%)")

            if args.output_json:
                with open(args.output_json, "w") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"\nSonuclar kaydedildi: {args.output_json}")
    else:
        print("HATA: --video veya --dir gerekli")
        parser.print_help()
        sys.exit(1)
