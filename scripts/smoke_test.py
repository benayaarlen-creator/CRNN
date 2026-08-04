from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crnn_multimodal.inference import analyze_video, load_model_bundle, self_test


MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "model_lr01f12_best.keras"
BASE_MODALITY_SCALE = 0.5
MFCC_MEAN_PATH = MODEL_DIR / "mfcc_train_mean.npy"
MFCC_STD_PATH = MODEL_DIR / "mfcc_train_std.npy"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke test aplikasi Streamlit CRNN Keras 8 kelas."
    )
    parser.add_argument("--video", type=Path)
    parser.add_argument("--segment-seconds", type=float, default=3.0)
    parser.add_argument("--audio-focus", type=float, default=0.5)
    args = parser.parse_args()

    started_at = time.perf_counter()
    bundle = load_model_bundle(
        MODEL_PATH,
        MFCC_MEAN_PATH,
        MFCC_STD_PATH,
        base_modality_scale=BASE_MODALITY_SCALE,
    )
    print(
        f"Model dimuat dalam {time.perf_counter() - started_at:.2f} detik",
        flush=True,
    )
    summary = self_test(bundle)
    print("MODEL SELF-TEST BERHASIL")
    for key, value in summary.items():
        print(f"{key}: {value}")

    if args.video is not None:
        video_started_at = time.perf_counter()

        def report_progress(done: int, total: int) -> None:
            print(
                f"Segmen {done}/{total} selesai setelah "
                f"{time.perf_counter() - video_started_at:.2f} detik",
                flush=True,
            )

        results, warnings = analyze_video(
            args.video,
            bundle,
            audio_focus=args.audio_focus,
            segment_seconds=args.segment_seconds,
            progress_callback=report_progress,
        )
        if not results:
            raise RuntimeError("Tidak ada hasil dari video smoke test.")
        print(f"VIDEO SELF-TEST BERHASIL: {len(results)} segmen")
        for result in results:
            print(
                f"{result.start_seconds:.1f}-{result.end_seconds:.1f}s: "
                f"{result.prediction.display_label} "
                f"({result.prediction.confidence * 100:.2f}%)"
            )
        for warning in warnings:
            print("PERINGATAN:", warning)


if __name__ == "__main__":
    main()
