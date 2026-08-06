from __future__ import annotations

import hashlib
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

RUNTIME_CACHE_DIR = PROJECT_ROOT / ".runtime_cache"
RUNTIME_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(RUNTIME_CACHE_DIR / "numba"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st

import crnn_multimodal.inference as inference
from crnn_multimodal.inference import (
    CLASS_DISPLAY_NAMES,
    CLASS_NAMES,
    DEFAULT_WINDOW_SECONDS,
    ModelBundle,
    Prediction,
    annotate_frame,
    load_model_bundle,
    measure_audio_quality,
    preprocess_audio,
    preprocess_frames,
)


MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "model_lr01f12_best.keras"
MFCC_MEAN_PATH = MODEL_DIR / "mfcc_train_mean.npy"
MFCC_STD_PATH = MODEL_DIR / "mfcc_train_std.npy"

SEGMENT_DURATION_OPTIONS = (1, 2, 3)
DETECTION_MODE_LABELS = {
    "Normal": "normal",
    "Jarak jauh (eksperimen)": "far",
}
MODEL_FUSION_SETTING = 0.5
CONFIDENCE_THRESHOLD = 0.45
MODEL_FRAME_COUNT = 12


@dataclass(frozen=True)
class AppSegmentPrediction:
    """Hasil prediksi satu potongan video."""

    start_seconds: float
    end_seconds: float
    prediction: Prediction
    face_detected: bool
    face_count: int
    preview_bgr: np.ndarray
    audio_used: bool

    def as_row(self) -> dict[str, object]:
        label = self.prediction.label
        display_label = CLASS_DISPLAY_NAMES.get(label, label)
        row: dict[str, object] = {
            "mulai_detik": round(self.start_seconds, 2),
            "selesai_detik": round(self.end_seconds, 2),
            "emosi": display_label,
            "kelas_model": label,
            "confidence_persen": round(self.prediction.confidence * 100.0, 2),
            "wajah_terdeteksi": self.face_detected,
            "frame_wajah": self.face_count,
            "audio_digunakan": self.audio_used,
        }
        for name, value in zip(CLASS_NAMES, self.prediction.probabilities):
            row[f"prob_{name}_persen"] = round(float(value) * 100.0, 2)
        return row


@st.cache_resource(show_spinner=False)
def get_model_bundle(
    model_path: str,
    model_modified_ns: int,
    mean_path: str,
    mean_modified_ns: int,
    std_path: str,
    std_modified_ns: int,
) -> ModelBundle:
    """Memuat model dan statistik normalisasi sekali."""
    del model_modified_ns, mean_modified_ns, std_modified_ns
    return load_model_bundle(
        model_path,
        mean_path,
        std_path,
        base_modality_scale=MODEL_FUSION_SETTING,
    )


def format_seconds(value: float) -> str:
    """Mengubah detik menjadi teks yang mudah dibaca."""
    minutes, seconds = divmod(value, 60.0)
    if minutes >= 1:
        return f"{int(minutes):02d}:{seconds:04.1f}"
    return f"{seconds:.1f} detik"


def enhance_frames_for_far_detection(
    frames: list[np.ndarray],
) -> list[np.ndarray]:
    """Memperjelas dan memperbesar frame agar wajah kecil lebih mudah dicari."""
    enhanced: list[np.ndarray] = []
    for frame in frames:
        height, width = frame.shape[:2]
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        lightness = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        ).apply(lightness)
        result = cv2.cvtColor(
            cv2.merge((lightness, channel_a, channel_b)),
            cv2.COLOR_LAB2BGR,
        )

        scale = min(2.0, max(1.0, 1280.0 / max(height, width)))
        if scale > 1.0:
            result = cv2.resize(
                result,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )
        enhanced.append(result)
    return enhanced


def sample_segment_frames(
    capture: cv2.VideoCapture,
    start_seconds: float,
    end_seconds: float,
) -> list[np.ndarray]:
    """Mengambil 12 frame secara merata dari satu segmen."""
    if end_seconds <= start_seconds:
        return []

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 25.0

    last_time = max(start_seconds, end_seconds - (1.0 / fps))
    sample_times = np.linspace(
        start_seconds,
        last_time,
        MODEL_FRAME_COUNT,
    )

    frames: list[np.ndarray] = []
    for timestamp in sample_times:
        capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
        ok, frame = capture.read()
        if ok and frame is not None:
            frames.append(frame)

    if not frames:
        return []

    while len(frames) < MODEL_FRAME_COUNT:
        frames.append(frames[-1].copy())
    return frames[:MODEL_FRAME_COUNT]


def extract_audio_track(
    video_path: Path,
    wav_path: Path,
) -> tuple[bool, str | None]:
    """Mengekstrak audio mono menggunakan fungsi proyek atau FFmpeg."""
    project_extractor = getattr(inference, "_extract_audio", None)
    if callable(project_extractor):
        try:
            return project_extractor(video_path, wav_path)
        except Exception as error:
            project_error = str(error)
    else:
        project_error = None

    sample_rate = int(getattr(inference, "SAMPLE_RATE", 22_050))
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0 and wav_path.is_file():
        return True, None

    message = completed.stderr.strip() or project_error or "Track audio tidak ditemukan."
    return False, message


def create_empty_audio_input(
    bundle: ModelBundle,
    sample_rate: int,
) -> np.ndarray:
    """Membuat input audio nol dengan bentuk yang diminta model."""
    duration = max(float(DEFAULT_WINDOW_SECONDS), 1.0)
    waveform = np.zeros(
        max(1, int(round(sample_rate * duration))),
        dtype=np.float32,
    )

    try:
        template = preprocess_audio(
            waveform,
            sample_rate,
            bundle.train_mean,
            bundle.train_std,
        )
        return np.zeros_like(template, dtype=np.float32)
    except Exception:
        pass

    for attribute in ("base_model", "model", "keras_model"):
        model = getattr(bundle, attribute, None)
        inputs = getattr(model, "inputs", None)
        if not inputs:
            continue
        for model_input in inputs:
            shape = tuple(
                int(value) if value is not None else 1
                for value in model_input.shape
            )
            input_name = str(getattr(model_input, "name", "")).lower()
            if "audio" in input_name or len(shape) == 4:
                return np.zeros(shape, dtype=np.float32)

    raise RuntimeError("Bentuk input audio model tidak dapat ditentukan.")


def audio_is_usable(
    waveform: np.ndarray,
    sample_rate: int,
) -> bool:
    """Menentukan apakah audio cukup terbaca untuk digunakan."""
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if not waveform.size or not np.isfinite(waveform).all():
        return False

    try:
        quality = measure_audio_quality(waveform, sample_rate)
        return bool(quality.has_usable_signal)
    except Exception:
        rms = float(
            np.sqrt(
                np.mean(
                    np.square(waveform, dtype=np.float64)
                )
            )
        )
        minimum_rms = float(getattr(inference, "MIN_AUDIO_RMS", 1e-4))
        return np.isfinite(rms) and rms >= minimum_rms


def analyze_video_recording(
    video_path: Path,
    bundle: ModelBundle,
    segment_seconds: int,
    detection_mode: str,
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[list[AppSegmentPrediction], list[str]]:
    """Menganalisis seluruh video, termasuk segmen dengan audio hening."""
    if segment_seconds not in SEGMENT_DURATION_OPTIONS:
        raise ValueError("Interval prediksi harus 1, 2, atau 3 detik.")
    if detection_mode not in DETECTION_MODE_LABELS.values():
        raise ValueError("Mode deteksi wajah tidak valid.")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Video tidak dapat dibuka: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or frame_count <= 0:
        capture.release()
        raise RuntimeError("Metadata durasi video tidak valid.")

    duration = frame_count / fps
    segment_count = max(
        1,
        int(math.ceil(duration / float(segment_seconds))),
    )

    sample_rate = int(getattr(inference, "SAMPLE_RATE", 22_050))
    empty_audio = create_empty_audio_input(bundle, sample_rate)

    results: list[AppSegmentPrediction] = []
    warnings: list[str] = []
    silent_audio_count = 0
    failed_frame_count = 0

    with tempfile.TemporaryDirectory(prefix="crnn8_audio_") as temp_dir:
        wav_path = Path(temp_dir) / "audio.wav"
        has_audio, audio_error = extract_audio_track(video_path, wav_path)
        audio_file = sf.SoundFile(wav_path) if has_audio else None

        try:
            for segment_index in range(segment_count):
                start = segment_index * float(segment_seconds)
                end = min(duration, start + float(segment_seconds))
                frames = sample_segment_frames(capture, start, end)

                if not frames:
                    failed_frame_count += 1
                    if progress_callback:
                        progress_callback(segment_index + 1, segment_count)
                    continue

                frames_for_model = (
                    enhance_frames_for_far_detection(frames)
                    if detection_mode == "far"
                    else frames
                )

                audio = empty_audio.copy()
                audio_used = False

                if audio_file is not None:
                    source_rate = int(audio_file.samplerate)
                    audio_file.seek(int(round(start * source_rate)))
                    requested_samples = max(
                        1,
                        int(math.ceil((end - start) * source_rate)),
                    )
                    waveform = audio_file.read(
                        frames=requested_samples,
                        dtype="float32",
                        always_2d=False,
                    )
                    waveform_array = np.asarray(
                        waveform,
                        dtype=np.float32,
                    ).reshape(-1)

                    if audio_is_usable(waveform_array, source_rate):
                        audio = preprocess_audio(
                            waveform_array,
                            source_rate,
                            bundle.train_mean,
                            bundle.train_std,
                        )
                        audio_used = True

                if not audio_used:
                    silent_audio_count += 1

                visual, visual_info = preprocess_frames(frames_for_model)
                face_count = int(
                    visual_info.get("face_detected_count", 0)
                    if isinstance(visual_info, dict)
                    else 0
                )

                prediction = bundle.predict(
                    visual,
                    audio,
                    MODEL_FUSION_SETTING,
                )

                preview = frames_for_model[len(frames_for_model) // 2]
                annotated, annotated_face = annotate_frame(
                    preview,
                    prediction,
                    confidence_threshold=CONFIDENCE_THRESHOLD,
                )

                results.append(
                    AppSegmentPrediction(
                        start_seconds=start,
                        end_seconds=end,
                        prediction=prediction,
                        face_detected=bool(annotated_face or face_count > 0),
                        face_count=face_count,
                        preview_bgr=annotated,
                        audio_used=audio_used,
                    )
                )

                if progress_callback:
                    progress_callback(segment_index + 1, segment_count)
        finally:
            if audio_file is not None:
                audio_file.close()
            capture.release()

    if silent_audio_count:
        warnings.append(
            f"{silent_audio_count} segmen memakai input audio nol karena suara hening atau tidak tersedia."
        )
    if not has_audio and audio_error:
        warnings.append("Track audio tidak digunakan: " + audio_error)
    if failed_frame_count:
        warnings.append(
            f"{failed_frame_count} segmen dilewati karena frame video gagal dibaca."
        )

    return results, warnings


def render_segment_preview(
    results: list[AppSegmentPrediction],
) -> None:
    """Menampilkan frame dari segmen yang dipilih."""
    choices = {
        (
            f"{format_seconds(item.start_seconds)} – "
            f"{format_seconds(item.end_seconds)} | "
            f"{item.prediction.display_label} "
            f"({item.prediction.confidence * 100:.1f}%)"
        ): index
        for index, item in enumerate(results)
    }

    selected_label = st.selectbox(
        "Pratinjau segmen",
        tuple(choices),
    )
    selected = results[choices[selected_label]]
    st.image(
        cv2.cvtColor(
            selected.preview_bgr,
            cv2.COLOR_BGR2RGB,
        ),
        caption=selected_label,
        width="stretch",
    )


def render_results(
    results: list[AppSegmentPrediction],
    warnings: list[str],
    segment_seconds: int,
) -> None:
    """Menampilkan ringkasan dan hasil setiap segmen."""
    if warnings:
        with st.expander("Catatan analisis"):
            for message in warnings:
                st.write(message)

    if not results:
        st.error("Tidak ada segmen video yang dapat dianalisis.")
        return

    table = pd.DataFrame(
        [result.as_row() for result in results]
    )
    dominant = table["emosi"].value_counts().index[0]
    average_confidence = float(
        table["confidence_persen"].mean()
    )

    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Segmen", len(table))
    metric_b.metric("Emosi dominan", str(dominant).upper())
    metric_c.metric(
        "Confidence rata-rata",
        f"{average_confidence:.1f}%",
    )

    st.subheader("Hasil prediksi")
    visible_columns = [
        "mulai_detik",
        "selesai_detik",
        "emosi",
        "confidence_persen",
        "frame_wajah",
        "audio_digunakan",
    ]
    st.dataframe(
        table[visible_columns],
        hide_index=True,
        width="stretch",
        column_config={
            "mulai_detik": st.column_config.NumberColumn(
                "Mulai",
                format="%.2f",
            ),
            "selesai_detik": st.column_config.NumberColumn(
                "Selesai",
                format="%.2f",
            ),
            "emosi": st.column_config.TextColumn("Emosi"),
            "confidence_persen": st.column_config.ProgressColumn(
                "Confidence",
                min_value=0.0,
                max_value=100.0,
                format="%.1f%%",
            ),
            "frame_wajah": st.column_config.NumberColumn(
                "Wajah terbaca",
                format="%d/12",
            ),
            "audio_digunakan": st.column_config.CheckboxColumn(
                "Audio",
            ),
        },
    )
    st.caption(
        "Audio yang tidak dicentang berarti segmen tetap diproses dengan input audio nol."
    )

    st.download_button(
        "Unduh CSV",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name=f"hasil_8_emosi_{segment_seconds}_detik.csv",
        mime="text/csv",
    )

    render_segment_preview(results)


def render_video_upload(
    bundle: ModelBundle,
    detection_mode: str,
    segment_seconds: int,
) -> None:
    """Menerima video rekaman dan menjalankan analisis."""
    uploaded = st.file_uploader(
        "Unggah video",
        type=["mp4", "mov", "avi", "mkv", "webm", "m4v"],
        max_upload_size=1024,
    )
    st.caption(
        "Audio hening tidak menghentikan analisis. Segmen tetap diprediksi menggunakan visual."
    )

    if uploaded is None:
        return

    video_bytes = uploaded.getvalue()
    st.video(video_bytes)

    signature = hashlib.sha256(video_bytes).hexdigest()
    run_signature = (
        f"{signature}:{detection_mode}:{segment_seconds}:"
        f"{MODEL_PATH.stat().st_mtime_ns}:silent-audio-v1"
    )

    if st.button(
        "Analisis",
        type="primary",
        width="stretch",
    ):
        progress = st.progress(
            0.0,
            text="Menyiapkan video...",
        )
        status = st.empty()

        def update_progress(done: int, total: int) -> None:
            progress.progress(
                done / max(1, total),
                text=f"Memproses {done}/{total}",
            )

        suffix = Path(uploaded.name).suffix or ".mp4"

        try:
            with tempfile.TemporaryDirectory(
                prefix="crnn8_upload_"
            ) as temp_dir:
                video_path = Path(temp_dir) / f"video{suffix}"
                video_path.write_bytes(video_bytes)
                results, warnings = analyze_video_recording(
                    video_path,
                    bundle,
                    segment_seconds,
                    detection_mode,
                    progress_callback=update_progress,
                )

            st.session_state.upload_analysis_8 = {
                "signature": run_signature,
                "results": results,
                "warnings": warnings,
            }
            progress.progress(1.0, text="Selesai")
            status.success("Analisis selesai.")
        except Exception as error:
            st.session_state.pop(
                "upload_analysis_8",
                None,
            )
            progress.empty()
            status.error(
                f"Gagal menganalisis video: {error}"
            )

    saved = st.session_state.get(
        "upload_analysis_8"
    )
    if saved:
        if saved["signature"] != run_signature:
            st.info(
                "Tekan Analisis untuk memperbarui hasil."
            )
        else:
            render_results(
                saved["results"],
                saved["warnings"],
                segment_seconds,
            )


def main() -> None:
    st.set_page_config(
        page_title="Deteksi Emosi Video – 8 Kelas",
        page_icon="🎭",
        layout="centered",
    )

    st.title("Deteksi Emosi Video")
    st.caption(
        "8 emosi • 12 frame • video rekaman"
    )

    st.subheader("Pengaturan")
    detection_column, interval_column = st.columns(2)

    with detection_column:
        detection_label = st.selectbox(
            "Deteksi wajah",
            tuple(DETECTION_MODE_LABELS),
            index=0,
        )
        st.caption(
            "Normal untuk wajah dekat. Jarak jauh membantu saat wajah terlihat lebih kecil."
        )
        detection_mode = DETECTION_MODE_LABELS[
            detection_label
        ]

    with interval_column:
        segment_seconds = st.selectbox(
            "Interval prediksi",
            SEGMENT_DURATION_OPTIONS,
            index=2,
            format_func=lambda value: f"{value} detik",
        )
        st.caption(
            "1 detik lebih rinci. 3 detik biasanya lebih stabil."
        )

    required_paths = (
        MODEL_PATH,
        MFCC_MEAN_PATH,
        MFCC_STD_PATH,
    )
    missing = [
        path
        for path in required_paths
        if not path.is_file()
    ]
    if missing:
        st.error(
            "Berkas model belum ditemukan: "
            + ", ".join(map(str, missing))
        )
        st.stop()

    try:
        with st.spinner("Memuat model..."):
            bundle = get_model_bundle(
                str(MODEL_PATH),
                MODEL_PATH.stat().st_mtime_ns,
                str(MFCC_MEAN_PATH),
                MFCC_MEAN_PATH.stat().st_mtime_ns,
                str(MFCC_STD_PATH),
                MFCC_STD_PATH.stat().st_mtime_ns,
            )
    except Exception as error:
        st.error(f"Model gagal dimuat: {error}")
        st.stop()

    render_video_upload(
        bundle,
        detection_mode,
        segment_seconds,
    )


if __name__ == "__main__":
    main()
