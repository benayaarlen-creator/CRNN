from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

RUNTIME_CACHE_DIR = PROJECT_ROOT / ".runtime_cache"
RUNTIME_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(RUNTIME_CACHE_DIR / "numba"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from crnn_multimodal.inference import (
    AudioQuality,
    CLASS_DISPLAY_NAMES,
    CLASS_NAMES,
    DEFAULT_WINDOW_SECONDS,
    ModelBundle,
    Prediction,
    SegmentPrediction,
    analyze_video,
    annotate_frame,
    audio_frame_to_mono,
    create_face_detector,
    detect_largest_face,
    empty_audio_input,
    load_model_bundle,
    merge_audio_chunks,
    measure_audio_quality,
    modality_scales,
    preprocess_audio,
    preprocess_frames,
)
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "model_lr01f12_best.keras"
MFCC_MEAN_PATH = MODEL_DIR / "mfcc_train_mean.npy"
MFCC_STD_PATH = MODEL_DIR / "mfcc_train_std.npy"
BASE_MODALITY_SCALE = 0.5


class LiveSession:
    """Thread-safe rolling audio/video buffers for an unlimited WebRTC session."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._detector_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="crnn-keras-live"
        )
        self._detector = create_face_detector()
        self._frames: list[np.ndarray] = []
        self._audio_chunks: list[tuple[np.ndarray, int]] = []
        self._window_started_at: float | None = None
        self._last_stored_frame_at = 0.0
        self._processing = False
        self._prediction: Prediction | None = None
        self._prediction_number = 0
        self._last_error: str | None = None
        self._last_traceback: str | None = None
        self._audio_focus = 0.30
        self._auto_quality = True
        self._confidence_threshold = 0.45
        self._smoothing_window = 3
        self._probability_history: list[np.ndarray] = []
        self._last_quality: dict[str, object] | None = None
        self._window_seconds = DEFAULT_WINDOW_SECONDS
        self._generation = 0
        self._overlay_face_box: tuple[int, int, int, int] | None = None
        self._last_overlay_detection_at = 0.0

    def configure(
        self,
        audio_focus: float,
        auto_quality: bool,
        confidence_threshold: float,
        smoothing_window: int,
    ) -> None:
        with self._lock:
            audio_focus = float(np.clip(audio_focus, 0.0, 1.0))
            smoothing_window = int(np.clip(smoothing_window, 1, 5))
            prediction_settings_changed = (
                not np.isclose(audio_focus, self._audio_focus)
                or bool(auto_quality) != self._auto_quality
                or smoothing_window != self._smoothing_window
            )
            self._audio_focus = audio_focus
            self._auto_quality = bool(auto_quality)
            self._confidence_threshold = float(
                np.clip(confidence_threshold, 0.0, 1.0)
            )
            self._smoothing_window = smoothing_window
            if prediction_settings_changed:
                self._probability_history.clear()

    def set_window_seconds(self, value: float) -> None:
        with self._lock:
            value = float(np.clip(value, 1.0, 5.0))
            if not np.isclose(value, self._window_seconds):
                self._window_seconds = value
                self._frames.clear()
                self._audio_chunks.clear()
                self._window_started_at = None
                self._probability_history.clear()
                self._generation += 1

    def reset(self) -> None:
        with self._lock:
            self._frames.clear()
            self._audio_chunks.clear()
            self._window_started_at = None
            self._last_stored_frame_at = 0.0
            self._prediction = None
            self._prediction_number = 0
            self._last_error = None
            self._last_traceback = None
            self._last_quality = None
            self._probability_history.clear()
            self._processing = False
            self._generation += 1

    def add_audio(self, frame: av.AudioFrame) -> None:
        try:
            chunk = audio_frame_to_mono(frame)
        except Exception as error:
            with self._lock:
                self._last_error = f"Audio live gagal dibaca: {error}"
                self._last_traceback = traceback.format_exc()
            return
        with self._lock:
            if self._window_started_at is None:
                return
            self._audio_chunks.append(chunk)
            if len(self._audio_chunks) > 1_000:
                self._audio_chunks = self._audio_chunks[-500:]

    def add_video(
        self,
        frame_bgr: np.ndarray,
        bundle: ModelBundle,
    ) -> None:
        now = time.monotonic()
        job: tuple[
            list[np.ndarray],
            list[tuple[np.ndarray, int]],
            float,
            bool,
            int,
            int,
        ] | None = None
        with self._lock:
            if self._window_started_at is None:
                self._window_started_at = now

            # Eight stored frames per second are enough for uniform sampling of
            # the 12 frames expected by the model.
            if now - self._last_stored_frame_at >= 0.125:
                self._frames.append(frame_bgr.copy())
                self._last_stored_frame_at = now

            elapsed = now - self._window_started_at
            if (
                elapsed >= self._window_seconds
                and not self._processing
                and self._frames
            ):
                job = (
                    self._frames,
                    self._audio_chunks,
                    self._audio_focus,
                    self._auto_quality,
                    self._smoothing_window,
                    self._generation,
                )
                self._frames = []
                self._audio_chunks = []
                self._window_started_at = now
                self._processing = True
                self._last_error = None
                self._last_traceback = None

        if job is not None:
            self._executor.submit(self._run_prediction, bundle, *job)

    def _run_prediction(
        self,
        bundle: ModelBundle,
        frames: list[np.ndarray],
        audio_chunks: list[tuple[np.ndarray, int]],
        audio_focus: float,
        auto_quality: bool,
        smoothing_window: int,
        generation: int,
    ) -> None:
        try:
            visual, visual_info = preprocess_frames(frames)
            face_count = int(visual_info["face_detected_count"])
            waveform = merge_audio_chunks(audio_chunks)
            audio_quality = measure_audio_quality(waveform, 22_050)
            effective_audio_focus = audio_focus
            if waveform.size and audio_quality.has_usable_signal:
                audio = preprocess_audio(
                    waveform,
                    22_050,
                    bundle.train_mean,
                    bundle.train_std,
                )
                if auto_quality and audio_quality.rms < 0.0020:
                    effective_audio_focus = min(audio_focus, 0.25)
                    audio_status = "Suara sangat pelan; bobot audio dikurangi."
                else:
                    audio_status = "Sinyal audio terdeteksi."
            else:
                audio = empty_audio_input()
                effective_audio_focus = 0.0
                audio_status = (
                    "Mikrofon hening/tidak aktif; prediksi memakai visual saja."
                )

            raw_prediction = bundle.predict(
                visual,
                audio,
                effective_audio_focus,
            )

            quality: dict[str, object] = {
                "face_count": face_count,
                "audio": audio_quality,
                "audio_status": audio_status,
                "requested_audio_focus": audio_focus,
                "effective_audio_focus": effective_audio_focus,
            }
            with self._lock:
                if generation == self._generation:
                    self._last_quality = quality

            if auto_quality and face_count < 3:
                raise RuntimeError(
                    f"Wajah hanya terdeteksi pada {face_count}/12 frame. "
                    "Hadapkan wajah ke kamera dan perbaiki pencahayaan."
                )

            with self._lock:
                if generation == self._generation:
                    self._probability_history.append(
                        raw_prediction.probabilities.copy()
                    )
                    self._probability_history = self._probability_history[
                        -smoothing_window:
                    ]
                    averaged = np.mean(
                        np.stack(self._probability_history), axis=0
                    ).astype(np.float32)
                    class_index = int(np.argmax(averaged))
                    prediction = Prediction(
                        label=CLASS_NAMES[class_index],
                        confidence=float(averaged[class_index]),
                        probabilities=averaged,
                    )
                    self._prediction = prediction
                    self._prediction_number += 1
                    self._last_error = None
                    self._last_traceback = None
        except Exception as error:
            error_traceback = traceback.format_exc()
            print(error_traceback, file=sys.stderr, flush=True)
            with self._lock:
                if generation == self._generation:
                    self._last_error = f"Prediksi live gagal: {error}"
                    self._last_traceback = error_traceback
        finally:
            with self._lock:
                if generation == self._generation:
                    self._processing = False

    def render(self, frame_bgr: np.ndarray) -> np.ndarray:
        now = time.monotonic()
        with self._lock:
            prediction = self._prediction
            processing = self._processing
            window_started_at = self._window_started_at
            window_seconds = self._window_seconds
            audio_focus = self._audio_focus
            confidence_threshold = self._confidence_threshold
            quality = self._last_quality
            should_detect = now - self._last_overlay_detection_at >= 0.30

        if should_detect:
            with self._detector_lock:
                face_box = detect_largest_face(
                    frame_bgr,
                    self._detector,
                    max_detection_side=480,
                )
            with self._lock:
                self._overlay_face_box = face_box
                self._last_overlay_detection_at = now

        with self._lock:
            face_box = self._overlay_face_box

        if prediction is None:
            if processing:
                status = "Memproses prediksi..."
            elif window_started_at is None:
                status = "Menunggu kamera..."
            else:
                remaining = max(
                    0.0, window_seconds - (now - window_started_at)
                )
                status = f"Mengumpulkan data {remaining:.1f}s"
        else:
            status = None

        annotated, _ = annotate_frame(
            frame_bgr,
            prediction,
            status_text=status,
            face_box=face_box,
            detect_face=False,
            confidence_threshold=confidence_threshold,
        )
        effective_audio_focus = (
            float(quality["effective_audio_focus"])
            if quality is not None
            else audio_focus
        )
        visual_percent = int(round((1.0 - effective_audio_focus) * 100.0))
        audio_percent = 100 - visual_percent
        focus_text = (
            f"Fokus efektif visual {visual_percent}% | suara {audio_percent}%"
        )
        cv2.putText(
            annotated,
            focus_text,
            (20, annotated.shape[0] - 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )
        return annotated

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            elapsed = (
                0.0
                if self._window_started_at is None
                else time.monotonic() - self._window_started_at
            )
            return {
                "prediction": self._prediction,
                "prediction_number": self._prediction_number,
                "processing": self._processing,
                "elapsed": elapsed,
                "window_seconds": self._window_seconds,
                "error": self._last_error,
                "traceback": self._last_traceback,
                "quality": self._last_quality,
                "confidence_threshold": self._confidence_threshold,
            }


@st.cache_resource(show_spinner=False)
def get_model_bundle(
    model_path: str,
    model_modified_ns: int,
    mean_path: str,
    mean_modified_ns: int,
    std_path: str,
    std_modified_ns: int,
    base_modality_scale: float,
) -> ModelBundle:
    del model_modified_ns, mean_modified_ns, std_modified_ns
    return load_model_bundle(
        model_path,
        mean_path,
        std_path,
        base_modality_scale=base_modality_scale,
    )


def get_live_session(model_path: Path) -> LiveSession:
    if "live_session" not in st.session_state:
        st.session_state.live_session = LiveSession()
        st.session_state.live_model_path = str(model_path)
    elif st.session_state.get("live_model_path") != str(model_path):
        st.session_state.live_session.reset()
        st.session_state.live_model_path = str(model_path)
    return st.session_state.live_session


def format_seconds(value: float) -> str:
    minutes, seconds = divmod(value, 60.0)
    if minutes >= 1:
        return f"{int(minutes):02d}:{seconds:04.1f}"
    return f"{seconds:.1f} detik"


def modality_controls(
    key_prefix: str,
    default_audio_percent: int = 50,
    base_scale: float = 1.0,
) -> float:
    audio_percent = st.slider(
        "Fokus visual atau suara",
        min_value=0,
        max_value=100,
        value=default_audio_percent,
        step=5,
        key=f"{key_prefix}_audio_focus",
        help=(
            "0 = visual saja, 50 = skala asli kedua cabang, dan 100 = suara saja. "
            "Pengaturan ini adalah bobot saat inferensi, bukan class_weight training."
        ),
    )
    audio_focus = audio_percent / 100.0
    visual_scale, audio_scale = modality_scales(audio_focus, base_scale)
    visual_column, audio_column = st.columns(2)
    visual_column.metric(
        "Visual", f"{100 - audio_percent}%", f"skala fitur {visual_scale:.2f}"
    )
    audio_column.metric(
        "Suara", f"{audio_percent}%", f"skala fitur {audio_scale:.2f}"
    )
    st.caption(
        "Posisi 50/50 mempertahankan konfigurasi model saat training. "
        f"Skala dasar model ini {base_scale:.2f} per cabang. "
        "Posisi lain meredam cabang yang tidak diprioritaskan."
    )
    return audio_focus


def render_probability_table(prediction: Prediction) -> None:
    probability_frame = pd.DataFrame(
        {
            "Emosi": [CLASS_DISPLAY_NAMES[name] for name in CLASS_NAMES],
            "Kelas model": CLASS_NAMES,
            "Probabilitas": prediction.probabilities,
        }
    ).sort_values("Probabilitas", ascending=False)
    st.dataframe(
        probability_frame,
        hide_index=True,
        width="stretch",
        column_config={
            "Probabilitas": st.column_config.ProgressColumn(
                "Probabilitas",
                min_value=0.0,
                max_value=1.0,
                format="percent",
            )
        },
    )


def render_live_tab(
    bundle: ModelBundle,
    model_path: Path,
) -> None:
    st.subheader("Prediksi langsung tanpa batas waktu")
    st.write(
        "Tekan **START**, lalu izinkan kamera dan mikrofon. "
        "Kotak wajah dan label emosi akan diperbarui berulang kali."
    )
    video_column, control_column = st.columns([2.25, 1.0], gap="large")
    state = get_live_session(model_path)

    with control_column:
        st.markdown("#### Pengaturan inferensi")
        audio_focus = modality_controls(
            "live",
            default_audio_percent=30,
            base_scale=bundle.base_modality_scale,
        )
        auto_quality = st.checkbox(
            "Sesuaikan dengan kualitas input",
            value=True,
            help=(
                "Jika mikrofon hening, cabang audio dinonaktifkan otomatis. "
                "Prediksi juga ditunda jika wajah tidak terlihat stabil."
            ),
        )
        window_seconds = st.slider(
            "Interval prediksi",
            min_value=1.0,
            max_value=5.0,
            value=DEFAULT_WINDOW_SECONDS,
            step=0.5,
            format="%.1f detik",
            help="Tiga detik adalah nilai bawaan yang aman untuk penggunaan live.",
        )
        confidence_threshold = st.slider(
            "Batas confidence",
            min_value=0.20,
            max_value=0.80,
            value=0.45,
            step=0.05,
            format="%.2f",
            help=(
                "Di bawah batas ini label ditampilkan sebagai Tidak yakin, "
                "bukan dipaksakan menjadi salah satu emosi."
            ),
        )
        smoothing_window = st.slider(
            "Perataan prediksi terakhir",
            min_value=1,
            max_value=5,
            value=3,
            step=1,
            help="Nilai 3 mengurangi perubahan label akibat satu jendela yang noisy.",
        )
        state.configure(
            audio_focus,
            auto_quality,
            confidence_threshold,
            smoothing_window,
        )
        state.set_window_seconds(window_seconds)
        if st.button("Reset hasil live", width="stretch"):
            state.reset()

        @st.fragment(run_every="1s")
        def live_status() -> None:
            snapshot = state.snapshot()
            error = snapshot["error"]
            prediction = snapshot["prediction"]
            quality = snapshot["quality"]
            if error:
                st.error(str(error))
                if snapshot["traceback"]:
                    with st.expander("Detail teknis error"):
                        st.code(str(snapshot["traceback"]), language=None)
            elif snapshot["processing"]:
                st.info("Model sedang memproses jendela terbaru...")
            elif prediction is None:
                remaining = max(
                    0.0,
                    float(snapshot["window_seconds"])
                    - float(snapshot["elapsed"]),
                )
                st.info(f"Menunggu prediksi pertama: {remaining:.1f} detik")
            else:
                assert isinstance(prediction, Prediction)
                message = (
                    f"Prediksi #{snapshot['prediction_number']}: "
                    f"{prediction.display_label.upper()} "
                    f"({prediction.confidence * 100:.1f}%)"
                )
                if prediction.confidence < float(
                    snapshot["confidence_threshold"]
                ):
                    st.warning(
                        "TIDAK YAKIN - kandidat tertinggi: " + message
                    )
                else:
                    st.success(message)
                render_probability_table(prediction)

            if quality is not None:
                audio_quality = quality["audio"]
                assert isinstance(audio_quality, AudioQuality)
                quality_a, quality_b = st.columns(2)
                quality_a.metric(
                    "Wajah terbaca",
                    f"{quality['face_count']}/12 frame",
                )
                quality_b.metric(
                    "Level suara",
                    f"{audio_quality.dbfs:.1f} dBFS",
                )
                requested = int(
                    round(float(quality["requested_audio_focus"]) * 100)
                )
                effective = int(
                    round(float(quality["effective_audio_focus"]) * 100)
                )
                st.caption(
                    f"{quality['audio_status']} Fokus audio diminta "
                    f"{requested}%, efektif {effective}%."
                )

        live_status()

    def video_callback(frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        state.add_video(image, bundle)
        result = state.render(image)
        return av.VideoFrame.from_ndarray(result, format="bgr24")

    def audio_callback(frame: av.AudioFrame) -> av.AudioFrame:
        state.add_audio(frame)
        return frame

    with video_column:
        webrtc_streamer(
            key="crnn-keras-live-camera-multimodal",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=video_callback,
            audio_frame_callback=audio_callback,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 960},
                    "height": {"ideal": 540},
                    "frameRate": {"ideal": 24, "max": 30},
                },
                "audio": True,
            },
            audio_html_attrs={"muted": True},
            async_processing=False,
        )
        st.caption(
            "Sesi tidak memiliki batas waktu. Audio keluaran dimatikan agar tidak "
            "memantul, tetapi mikrofon tetap dipakai oleh model."
        )


def render_segment_preview(results: list[SegmentPrediction]) -> None:
    choices = {
        f"{format_seconds(item.start_seconds)} - {format_seconds(item.end_seconds)} | "
        f"{item.prediction.display_label} "
        f"({item.prediction.confidence * 100:.1f}%)": index
        for index, item in enumerate(results)
    }
    selected_label = st.selectbox("Lihat frame representatif", tuple(choices))
    selected = results[choices[selected_label]]
    st.image(
        cv2.cvtColor(selected.preview_bgr, cv2.COLOR_BGR2RGB),
        caption=selected_label,
        width="stretch",
    )


def render_upload_results(
    results: list[SegmentPrediction],
    warnings: list[str],
) -> None:
    if not results:
        st.error("Tidak ada segmen video yang berhasil dianalisis.")
        return
    for message in warnings:
        st.warning(message)

    table = pd.DataFrame([result.as_row() for result in results])
    dominant = table["emosi"].value_counts().index[0]
    average_confidence = float(table["confidence_persen"].mean())
    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Jumlah segmen", len(table))
    metric_b.metric("Emosi dominan", str(dominant).upper())
    metric_c.metric("Rata-rata confidence", f"{average_confidence:.1f}%")

    st.markdown("#### Timeline emosi")
    visible_columns = [
        "mulai_detik",
        "selesai_detik",
        "emosi",
        "kelas_model",
        "confidence_persen",
        "wajah_terdeteksi",
    ]
    st.dataframe(
        table[visible_columns],
        hide_index=True,
        width="stretch",
        column_config={
            "mulai_detik": st.column_config.NumberColumn(
                "Mulai (detik)", format="%.2f"
            ),
            "selesai_detik": st.column_config.NumberColumn(
                "Selesai (detik)", format="%.2f"
            ),
            "emosi": st.column_config.TextColumn("Emosi"),
            "kelas_model": st.column_config.TextColumn("Kelas model"),
            "confidence_persen": st.column_config.ProgressColumn(
                "Confidence",
                min_value=0.0,
                max_value=100.0,
                format="%.1f%%",
            ),
            "wajah_terdeteksi": st.column_config.CheckboxColumn("Wajah"),
        },
    )

    st.markdown("#### Distribusi segmen")
    counts = (
        table["emosi"]
        .value_counts()
        .rename_axis("Emosi")
        .reset_index(name="Jumlah segmen")
    )
    st.bar_chart(counts, x="Emosi", y="Jumlah segmen", color="#5B5BD6")

    st.download_button(
        "Unduh hasil lengkap (CSV)",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="timeline_emosi_crnn_8_kelas.csv",
        mime="text/csv",
    )
    render_segment_preview(results)


def render_upload_tab(bundle: ModelBundle, model_path: Path) -> None:
    st.subheader("Analisis video berdasarkan interval waktu")
    st.write(
        "Unggah video; aplikasi akan membaginya menjadi beberapa segmen dan "
        "menampilkan emosi yang terdeteksi pada setiap rentang detik."
    )
    input_column, settings_column = st.columns([2.0, 1.0], gap="large")
    with settings_column:
        st.markdown("#### Pengaturan analisis")
        audio_focus = modality_controls(
            "upload",
            base_scale=bundle.base_modality_scale,
        )
        segment_seconds = st.slider(
            "Durasi setiap segmen",
            min_value=1.0,
            max_value=5.0,
            value=DEFAULT_WINDOW_SECONDS,
            step=0.5,
            format="%.1f detik",
            help="Model akan mengambil 12 frame merata dari setiap segmen.",
        )
        st.info(
            "Nilai 3 detik direkomendasikan. Segmen lebih pendek akan dipadding "
            "dan segmen terakhir tetap dianalisis."
        )

    with input_column:
        uploaded = st.file_uploader(
            "Pilih video",
            type=["mp4", "mov", "avi", "mkv", "webm", "m4v"],
            max_upload_size=500,
            help="Batas unggahan aplikasi adalah 500 MB per video.",
        )
        if uploaded is not None:
            video_bytes = uploaded.getvalue()
            st.video(video_bytes)
            signature = hashlib.sha256(video_bytes).hexdigest()
            run_signature = (
                f"{signature}:{model_path}:{audio_focus:.2f}:"
                f"{segment_seconds:.1f}"
            )
            if st.button("Analisis video", type="primary", width="stretch"):
                progress = st.progress(0.0, text="Menyiapkan video...")
                status = st.empty()

                def update_progress(done: int, total: int) -> None:
                    progress.progress(
                        done / max(1, total),
                        text=f"Menganalisis segmen {done} dari {total}...",
                    )

                suffix = Path(uploaded.name).suffix or ".mp4"
                try:
                    with tempfile.TemporaryDirectory(
                        prefix="crnn_keras_upload_"
                    ) as temp_dir:
                        video_path = Path(temp_dir) / f"video{suffix}"
                        video_path.write_bytes(video_bytes)
                        results, warnings = analyze_video(
                            video_path,
                            bundle,
                            audio_focus,
                            segment_seconds=segment_seconds,
                            progress_callback=update_progress,
                        )
                    st.session_state.upload_analysis_keras = {
                        "signature": run_signature,
                        "results": results,
                        "warnings": warnings,
                    }
                    progress.progress(1.0, text="Analisis selesai.")
                    status.success(
                        "Semua segmen yang dapat dibaca sudah dianalisis."
                    )
                except Exception as error:
                    st.session_state.pop("upload_analysis_keras", None)
                    progress.empty()
                    status.error(f"Analisis video gagal: {error}")

            saved = st.session_state.get("upload_analysis_keras")
            if saved:
                if saved["signature"] != run_signature:
                    st.warning(
                        "Video, model, durasi segmen, atau bobot modalitas berubah. "
                        "Tekan **Analisis video** untuk memperbarui hasil."
                    )
                else:
                    render_upload_results(saved["results"], saved["warnings"])


def render_model_info(model_path: Path, bundle: ModelBundle) -> None:
    st.subheader("Informasi model yang sedang dipakai")
    st.code(str(model_path), language=None)
    columns = st.columns(4)
    columns[0].metric("Kelas emosi", len(CLASS_NAMES))
    columns[1].metric("Frame visual", "12")
    columns[2].metric("Ukuran wajah", "64 x 64 RGB")
    columns[3].metric("Input audio", "40 x 531 MFCC")
    st.caption(
        "Skala modalitas pada posisi 50/50: "
        f"visual {bundle.base_modality_scale:.2f}, "
        f"audio {bundle.base_modality_scale:.2f}."
    )
    st.markdown(
        """
        Model menggunakan dua cabang CNN-LSTM: visual dan audio. Fitur kedua
        cabang digabung dengan **model-level fusion**, lalu diklasifikasikan ke
        delapan kelas RAVDESS. Statistik normalisasi audio berasal dari split
        training yang sama dengan model.

        Model ini dilatih pada ekspresi terkontrol RAVDESS. Hasil live di dunia
        nyata dapat berbeda karena pencahayaan, posisi wajah, kualitas mikrofon,
        bahasa, dan karakter suara. Gunakan hasil sebagai keluaran penelitian,
        bukan diagnosis psikologis.
        """
    )


def main() -> None:
    st.set_page_config(
        page_title="CRNN Multimodal 8 Emosi",
        page_icon="🎭",
        layout="wide",
    )
    st.title("Deteksi Emosi Multimodal CRNN - 8 Kelas")
    st.caption(
        "Live webcam + mikrofon dan analisis video menggunakan model Keras "
        "model-level fusion"
    )

    with st.sidebar:
        st.header("Model Keras")
        model_path = MODEL_PATH
        st.code(str(model_path), language=None)
        st.caption("Model tunggal: LR 0.0001, 12 frame, test accuracy 75%")
        st.caption(
            "Kelas: "
            + ", ".join(CLASS_DISPLAY_NAMES[name] for name in CLASS_NAMES)
        )

    required_paths = (
        model_path,
        MFCC_MEAN_PATH,
        MFCC_STD_PATH,
    )
    missing = [path for path in required_paths if not path.is_file()]
    if missing:
        st.error("Berkas yang diperlukan belum ditemukan: " + ", ".join(map(str, missing)))
        st.stop()

    try:
        with st.spinner("Memuat model CRNN multimodal 8 kelas..."):
            bundle = get_model_bundle(
                str(model_path),
                model_path.stat().st_mtime_ns,
                str(MFCC_MEAN_PATH),
                MFCC_MEAN_PATH.stat().st_mtime_ns,
                str(MFCC_STD_PATH),
                MFCC_STD_PATH.stat().st_mtime_ns,
                BASE_MODALITY_SCALE,
            )
    except Exception as error:
        st.error(f"Model gagal dimuat: {error}")
        st.stop()

    with st.sidebar:
        st.success("Model siap digunakan")
        st.write(f"Ukuran: {model_path.stat().st_size / (1024 * 1024):.2f} MB")
        st.warning(
            "Slider fokus adalah eksperimen bobot inferensi, bukan class_weight "
            "training. Nilai evaluasi model berlaku pada posisi asli 50/50."
        )

    live_tab, upload_tab, info_tab = st.tabs(
        ["Kamera langsung", "Unggah video", "Informasi model"]
    )
    with live_tab:
        render_live_tab(bundle, model_path)
    with upload_tab:
        render_upload_tab(bundle, model_path)
    with info_tab:
        render_model_info(model_path, bundle)


if __name__ == "__main__":
    main()
