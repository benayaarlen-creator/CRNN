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

# Nilai ini tetap dipakai secara internal agar perilaku model sama dengan versi asli.
# Nilainya tidak ditampilkan di antarmuka.
MODEL_MODALITY_SCALE = 0.5
SEGMENT_DURATION_OPTIONS = (1, 2, 3)


@st.cache_resource(show_spinner=False)
def get_model_bundle(
    model_path: str,
    model_modified_ns: int,
    mean_path: str,
    mean_modified_ns: int,
    std_path: str,
    std_modified_ns: int,
) -> ModelBundle:
    """Memuat model dan statistik normalisasi tanpa mengubah konfigurasi model."""
    del model_modified_ns, mean_modified_ns, std_modified_ns
    return load_model_bundle(
        model_path,
        mean_path,
        std_path,
        base_modality_scale=MODEL_MODALITY_SCALE,
    )


def format_seconds(value: float) -> str:
    minutes, seconds = divmod(value, 60.0)
    if minutes >= 1:
        return f"{int(minutes):02d}:{seconds:04.1f}"
    return f"{seconds:.1f} detik"


def render_segment_preview(results: list[SegmentPrediction]) -> None:
    choices = {
        (
            f"{format_seconds(item.start_seconds)} – "
            f"{format_seconds(item.end_seconds)} | "
            f"{item.prediction.display_label} "
            f"({item.prediction.confidence * 100:.1f}%)"
        ): index
        for index, item in enumerate(results)
    }

    selected_label = st.selectbox("Pratinjau segmen", tuple(choices))
    selected = results[choices[selected_label]]
    st.image(
        cv2.cvtColor(selected.preview_bgr, cv2.COLOR_BGR2RGB),
        caption=selected_label,
        width="stretch",
    )


def render_results(
    results: list[SegmentPrediction],
    warnings: list[str],
    segment_seconds: int,
) -> None:
    if warnings:
        with st.expander("Catatan analisis"):
            for message in warnings:
                st.write(message)

    if not results:
        st.error("Tidak ada segmen video yang berhasil dianalisis.")
        return

    table = pd.DataFrame([result.as_row() for result in results])
    dominant = table["emosi"].value_counts().index[0]
    average_confidence = float(table["confidence_persen"].mean())

    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Segmen", len(table))
    metric_b.metric("Emosi dominan", str(dominant).upper())
    metric_c.metric("Confidence rata-rata", f"{average_confidence:.1f}%")

    st.subheader("Hasil prediksi")
    visible_columns = [
        "mulai_detik",
        "selesai_detik",
        "emosi",
        "kelas_model",
        "confidence_persen",
        "wajah_terdeteksi",
    ]
    visible_columns = [column for column in visible_columns if column in table.columns]

    column_config: dict[str, object] = {
        "mulai_detik": st.column_config.NumberColumn("Mulai", format="%.2f"),
        "selesai_detik": st.column_config.NumberColumn("Selesai", format="%.2f"),
        "emosi": st.column_config.TextColumn("Emosi"),
        "kelas_model": st.column_config.TextColumn("Kelas model"),
        "confidence_persen": st.column_config.ProgressColumn(
            "Confidence",
            min_value=0.0,
            max_value=100.0,
            format="%.1f%%",
        ),
        "wajah_terdeteksi": st.column_config.CheckboxColumn("Wajah"),
    }

    st.dataframe(
        table[visible_columns],
        hide_index=True,
        width="stretch",
        column_config={
            key: value
            for key, value in column_config.items()
            if key in visible_columns
        },
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
    segment_seconds: int,
) -> None:
    uploaded = st.file_uploader(
        "Unggah video",
        type=["mp4", "mov", "avi", "mkv", "webm", "m4v"],
        max_upload_size=500,
    )

    if uploaded is None:
        return

    video_bytes = uploaded.getvalue()
    st.video(video_bytes)

    signature = hashlib.sha256(video_bytes).hexdigest()
    run_signature = (
        f"{signature}:{segment_seconds}:"
        f"{MODEL_PATH.stat().st_mtime_ns}:original-pipeline-v1"
    )

    if st.button("Analisis", type="primary", width="stretch"):
        progress = st.progress(0.0, text="Menyiapkan video...")
        status = st.empty()

        def update_progress(done: int, total: int) -> None:
            progress.progress(
                done / max(1, total),
                text=f"Memproses {done}/{total}",
            )

        suffix = Path(uploaded.name).suffix or ".mp4"

        try:
            with tempfile.TemporaryDirectory(prefix="crnn8_upload_") as temp_dir:
                video_path = Path(temp_dir) / f"video{suffix}"
                video_path.write_bytes(video_bytes)

                # Memakai fungsi analisis asli proyek supaya sampling frame,
                # deteksi wajah, MFCC, normalisasi, dan fusion tetap sama.
                results, warnings = analyze_video(
                    video_path,
                    bundle,
                    MODEL_MODALITY_SCALE,
                    segment_seconds=float(segment_seconds),
                    progress_callback=update_progress,
                )

            st.session_state.upload_analysis_8_stable = {
                "signature": run_signature,
                "results": results,
                "warnings": warnings,
            }
            progress.progress(1.0, text="Selesai")
            status.success("Analisis selesai.")
        except Exception as error:
            st.session_state.pop("upload_analysis_8_stable", None)
            progress.empty()
            status.error(f"Gagal menganalisis video: {error}")

    saved = st.session_state.get("upload_analysis_8_stable")
    if saved:
        if saved["signature"] != run_signature:
            st.info("Tekan Analisis untuk memperbarui hasil.")
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
    st.caption("8 emosi • 12 frame • video rekaman")

    st.subheader("Pengaturan")
    info_column, interval_column = st.columns(2)

    with info_column:
        st.markdown("**Deteksi wajah**")
        st.caption(
            "Dilakukan otomatis oleh preprocessing asli model agar hasil tetap konsisten."
        )

    with interval_column:
        segment_seconds = st.selectbox(
            "Interval prediksi",
            SEGMENT_DURATION_OPTIONS,
            index=2,
            format_func=lambda value: f"{value} detik",
        )
        st.caption("3 detik paling mendekati konfigurasi aplikasi asli.")

    required_paths = (
        MODEL_PATH,
        MFCC_MEAN_PATH,
        MFCC_STD_PATH,
    )
    missing = [path for path in required_paths if not path.is_file()]
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

    render_video_upload(bundle, segment_seconds)


if __name__ == "__main__":
    main()
