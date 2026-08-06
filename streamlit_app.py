from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

RUNTIME_CACHE_DIR = PROJECT_ROOT / ".runtime_cache"
RUNTIME_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(RUNTIME_CACHE_DIR / "numba"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import pandas as pd
import streamlit as st

from crnn_multimodal.inference import (
    CLASS_DISPLAY_NAMES,
    CLASS_NAMES,
    ModelBundle,
    SegmentPrediction,
    analyze_video,
    load_model_bundle,
)

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "model_lr01f12_best.keras"
MFCC_MEAN_PATH = MODEL_DIR / "mfcc_train_mean.npy"
MFCC_STD_PATH = MODEL_DIR / "mfcc_train_std.npy"

# Tetap dipakai secara internal agar cara kerja model tidak berubah.
BASE_MODALITY_SCALE = 0.5
INFERENCE_AUDIO_FOCUS = 0.5
SEGMENT_DURATION_OPTIONS = (1.0, 2.0, 3.0)


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


def format_seconds(value: float) -> str:
    minutes, seconds = divmod(value, 60.0)
    if minutes >= 1:
        return f"{int(minutes):02d}:{seconds:04.1f}"
    return f"{seconds:.1f} detik"


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
    segment_seconds: float,
) -> None:
    if not results:
        st.error("Tidak ada segmen video yang berhasil dianalisis.")
        return

    if warnings:
        with st.expander("Peringatan analisis"):
            for message in warnings:
                st.write(message)

    table = pd.DataFrame([result.as_row() for result in results])
    dominant = table["emosi"].value_counts().index[0]
    average_confidence = float(table["confidence_persen"].mean())

    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Jumlah segmen", len(table))
    metric_b.metric("Emosi dominan", str(dominant).upper())
    metric_c.metric("Rata-rata confidence", f"{average_confidence:.1f}%")

    st.subheader("Timeline emosi")
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

    st.subheader("Distribusi segmen")
    counts = (
        table["emosi"]
        .value_counts()
        .rename_axis("Emosi")
        .reset_index(name="Jumlah segmen")
    )
    st.bar_chart(counts, x="Emosi", y="Jumlah segmen")

    st.download_button(
        "Unduh hasil lengkap (CSV)",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name=f"timeline_emosi_8_kelas_{int(segment_seconds)}_detik.csv",
        mime="text/csv",
    )

    render_segment_preview(results)


def render_upload_tab(bundle: ModelBundle, model_path: Path) -> None:
    st.subheader("Analisis video rekaman")

    segment_seconds = st.selectbox(
        "Interval prediksi",
        SEGMENT_DURATION_OPTIONS,
        index=2,
        format_func=lambda value: f"{int(value)} detik",
        help=(
            "Video dibagi menjadi potongan 1, 2, atau 3 detik. "
            "Pilihan 3 detik paling dekat dengan konfigurasi awal model."
        ),
    )

    uploaded = st.file_uploader(
        "Pilih video",
        type=["mp4", "mov", "avi", "mkv", "webm", "m4v"],
        max_upload_size=500,
        help="Batas unggahan adalah 500 MB per video.",
    )

    if uploaded is None:
        st.info("Unggah satu video untuk memulai analisis.")
        return

    video_bytes = uploaded.getvalue()
    st.video(video_bytes)

    signature = hashlib.sha256(video_bytes).hexdigest()
    run_signature = (
        f"{signature}:{model_path}:{INFERENCE_AUDIO_FOCUS:.2f}:"
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

                # Menggunakan fungsi analisis asli agar preprocessing tetap sama.
                results, warnings = analyze_video(
                    video_path,
                    bundle,
                    INFERENCE_AUDIO_FOCUS,
                    segment_seconds=segment_seconds,
                    progress_callback=update_progress,
                )

            st.session_state.upload_analysis_keras = {
                "signature": run_signature,
                "results": results,
                "warnings": warnings,
                "segment_seconds": segment_seconds,
            }
            progress.progress(1.0, text="Analisis selesai.")
            status.success("Analisis selesai.")
        except Exception as error:
            st.session_state.pop("upload_analysis_keras", None)
            progress.empty()
            status.error(f"Analisis video gagal: {error}")

    saved = st.session_state.get("upload_analysis_keras")
    if saved:
        if saved["signature"] != run_signature:
            st.warning(
                "Video atau interval berubah. Tekan **Analisis video** "
                "untuk memperbarui hasil."
            )
        else:
            render_upload_results(
                saved["results"],
                saved["warnings"],
                float(saved["segment_seconds"]),
            )


def render_model_info(model_path: Path) -> None:
    st.subheader("Informasi model")
    st.code(str(model_path), language=None)

    columns = st.columns(4)
    columns[0].metric("Kelas emosi", len(CLASS_NAMES))
    columns[1].metric("Frame visual", "12")
    columns[2].metric("Ukuran wajah", "64 x 64 RGB")
    columns[3].metric("Input audio", "40 x 531 MFCC")

    st.write(
        "Model menggunakan cabang visual dan audio berbasis CNN-LSTM. "
        "Fitur kedua cabang digabungkan sebelum klasifikasi ke delapan kelas emosi."
    )


def main() -> None:
    st.set_page_config(
        page_title="CRNN Multimodal 8 Emosi",
        page_icon="🎭",
        layout="wide",
    )

    st.title("Deteksi Emosi Multimodal CRNN - 8 Kelas")
    st.caption("Analisis emosi dari video yang sudah direkam.")

    with st.sidebar:
        st.header("Model")
        model_path = MODEL_PATH
        st.caption("12 frame · 8 kelas emosi")
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
        st.error(
            "Berkas yang diperlukan belum ditemukan: "
            + ", ".join(map(str, missing))
        )
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

    upload_tab, info_tab = st.tabs(["Unggah video", "Informasi model"])
    with upload_tab:
        render_upload_tab(bundle, model_path)
    with info_tab:
        render_model_info(model_path)


if __name__ == "__main__":
    main()
