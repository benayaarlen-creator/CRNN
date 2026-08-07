from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import cv2
import h5py
import imageio_ffmpeg
import numpy as np
import soundfile as sf
import tensorflow as tf
from scipy import fft as scipy_fft
from scipy import signal as scipy_signal
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers


CLASS_NAMES = (
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
)

CLASS_DISPLAY_NAMES = {
    "neutral": "Netral",
    "calm": "Tenang",
    "happy": "Senang",
    "sad": "Sedih",
    "angry": "Marah",
    "fearful": "Takut",
    "disgust": "Jijik",
    "surprised": "Terkejut",
}

# OpenCV uses BGR colors.
CLASS_COLORS = {
    "neutral": (224, 224, 224),
    "calm": (205, 172, 68),
    "happy": (35, 204, 235),
    "sad": (235, 130, 48),
    "angry": (52, 52, 235),
    "fearful": (186, 85, 211),
    "disgust": (78, 168, 84),
    "surprised": (70, 150, 255),
}

NUM_FRAMES = 12
IMAGE_SIZE = 64
SAMPLE_RATE = 22_050
DEFAULT_WINDOW_SECONDS = 3.0
MIN_RELIABLE_SEGMENT_SECONDS = 2.0
N_MFCC = 40
TARGET_TIME_STEPS = 531
AUDIO_SHAPE = (N_MFCC, TARGET_TIME_STEPS, 1)
FRAME_LENGTH_MS = 25
HOP_LENGTH_MS = 10
WIN_LENGTH = round(SAMPLE_RATE * FRAME_LENGTH_MS / 1000)
HOP_LENGTH = round(SAMPLE_RATE * HOP_LENGTH_MS / 1000)
N_FFT = 1024
N_MELS = 128
L2_FACTOR = 1e-4


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    probabilities: np.ndarray

    @property
    def display_label(self) -> str:
        return CLASS_DISPLAY_NAMES[self.label]


@dataclass(frozen=True)
class AudioQuality:
    rms: float
    peak: float
    dbfs: float
    duration_seconds: float
    has_usable_signal: bool


@dataclass(frozen=True)
class SegmentPrediction:
    start_seconds: float
    end_seconds: float
    prediction: Prediction
    face_detected: bool
    preview_bgr: np.ndarray

    def as_row(self) -> dict[str, object]:
        row: dict[str, object] = {
            "mulai_detik": round(self.start_seconds, 2),
            "selesai_detik": round(self.end_seconds, 2),
            "emosi": self.prediction.display_label,
            "kelas_model": self.prediction.label,
            "confidence_persen": round(self.prediction.confidence * 100.0, 2),
            "wajah_terdeteksi": self.face_detected,
        }
        for name, value in zip(CLASS_NAMES, self.prediction.probabilities):
            row[f"prob_{name}_persen"] = round(float(value) * 100.0, 2)
        return row


@dataclass
class ModelBundle:
    base_model: tf.keras.Model
    encoder: tf.keras.Model
    fusion_head: tf.keras.Model
    inference_function: Callable[..., tf.Tensor]
    train_mean: np.ndarray
    train_std: np.ndarray
    model_path: Path
    base_modality_scale: float
    lock: threading.Lock

    def predict(
        self,
        visual: np.ndarray,
        audio: np.ndarray,
        audio_focus: float = 0.5,
    ) -> Prediction:
        visual_scale, audio_scale = modality_scales(
            audio_focus,
            self.base_modality_scale,
        )
        with self.lock:
            probabilities = self.inference_function(
                tf.convert_to_tensor(visual, dtype=tf.float32),
                tf.convert_to_tensor(audio, dtype=tf.float32),
                tf.constant(visual_scale, dtype=tf.float32),
                tf.constant(audio_scale, dtype=tf.float32),
            ).numpy()[0]

        class_index = int(np.argmax(probabilities))
        return Prediction(
            label=CLASS_NAMES[class_index],
            confidence=float(probabilities[class_index]),
            probabilities=np.asarray(probabilities, dtype=np.float32),
        )


def modality_scales(
    audio_focus: float,
    base_scale: float = 1.0,
) -> tuple[float, float]:
    """Map a visual/audio ratio to feature attenuation at inference time.

    The midpoint keeps both branches at their trained scale. Moving away from
    it attenuates only the less-focused branch, so neither feature branch is
    amplified beyond the range seen during training.
    """
    if not 0.0 <= audio_focus <= 1.0:
        raise ValueError("audio_focus harus berada pada rentang 0.0 sampai 1.0")
    if not 0.0 < base_scale <= 1.0:
        raise ValueError("base_scale harus lebih dari 0 dan maksimal 1")
    if np.isclose(audio_focus, 0.5):
        return base_scale, base_scale
    if audio_focus < 0.5:
        return base_scale, base_scale * audio_focus / (1.0 - audio_focus)
    return base_scale * (1.0 - audio_focus) / audio_focus, base_scale


def _visual_conv_block(
    inputs: tf.Tensor,
    filters: int,
    block_number: int,
) -> tf.Tensor:
    prefix = f"visual_block{block_number}"
    value = layers.TimeDistributed(
        layers.Conv2D(
            filters,
            kernel_size=(3, 3),
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizers.l2(L2_FACTOR),
        ),
        name=f"{prefix}_conv",
    )(inputs)
    value = layers.TimeDistributed(
        layers.BatchNormalization(), name=f"{prefix}_batchnorm"
    )(value)
    value = layers.TimeDistributed(
        layers.Activation("relu"), name=f"{prefix}_relu"
    )(value)
    return layers.TimeDistributed(
        layers.MaxPooling2D(pool_size=(2, 2)), name=f"{prefix}_pool"
    )(value)


def _audio_conv_block(
    inputs: tf.Tensor,
    filters: int,
    block_number: int,
) -> tf.Tensor:
    prefix = f"audio_block{block_number}"
    value = layers.Conv2D(
        filters,
        kernel_size=(3, 3),
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(L2_FACTOR),
        name=f"{prefix}_conv",
    )(inputs)
    value = layers.BatchNormalization(name=f"{prefix}_batchnorm")(value)
    value = layers.Activation("relu", name=f"{prefix}_relu")(value)
    return layers.MaxPooling2D(
        pool_size=(2, 2), name=f"{prefix}_pool"
    )(value)


def build_model() -> tf.keras.Model:
    """Rebuild the exact Keras 2.10 architecture saved in the supplied files."""
    visual_input = keras.Input(
        shape=(NUM_FRAMES, IMAGE_SIZE, IMAGE_SIZE, 3), name="visual_input"
    )
    visual = visual_input
    for block_number, filters in enumerate((32, 64, 128), start=1):
        visual = _visual_conv_block(visual, filters, block_number)
    visual = layers.TimeDistributed(
        layers.GlobalAveragePooling2D(),
        name="visual_global_average_pooling",
    )(visual)
    visual = layers.LSTM(128, dropout=0.25, name="visual_lstm")(visual)
    visual = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(L2_FACTOR),
        name="visual_features",
    )(visual)
    visual = layers.Dropout(0.30, name="visual_dropout")(visual)

    audio_input = keras.Input(shape=AUDIO_SHAPE, name="audio_input")
    audio = audio_input
    for block_number, filters in enumerate((32, 64, 128), start=1):
        audio = _audio_conv_block(audio, filters, block_number)
    audio = layers.Permute((2, 1, 3), name="audio_time_first")(audio)
    audio = layers.TimeDistributed(
        layers.Flatten(), name="audio_flatten_per_timestep"
    )(audio)
    audio = layers.LSTM(
        128,
        dropout=0.25,
        recurrent_dropout=0.0,
        name="audio_lstm",
    )(audio)
    audio = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(L2_FACTOR),
        name="audio_features",
    )(audio)
    audio = layers.Dropout(0.30, name="audio_dropout")(audio)

    fusion = layers.Concatenate(name="model_level_fusion")([visual, audio])
    fusion = layers.Dense(
        256,
        use_bias=False,
        kernel_regularizer=regularizers.l2(L2_FACTOR),
        name="fusion_dense_256",
    )(fusion)
    fusion = layers.BatchNormalization(name="fusion_batchnorm")(fusion)
    fusion = layers.Activation("relu", name="fusion_relu_256")(fusion)
    fusion = layers.Dropout(0.40, name="fusion_dropout_40")(fusion)
    fusion = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(L2_FACTOR),
        name="fusion_dense_128",
    )(fusion)
    fusion = layers.Dropout(0.30, name="fusion_dropout_30")(fusion)
    output = layers.Dense(
        len(CLASS_NAMES), activation="softmax", name="emotion_output"
    )(fusion)

    return keras.Model(
        inputs={"visual_input": visual_input, "audio_input": audio_input},
        outputs=output,
        name="CRNN_Model_Level_Fusion_8Class",
    )


def _load_weights_compatible(model: tf.keras.Model, model_path: Path) -> None:
    if not model_path.is_file():
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")
    if not h5py.is_hdf5(model_path):
        raise ValueError(f"Model bukan HDF5 Keras yang didukung: {model_path}")

    # The supplied files use the .keras suffix but contain legacy Keras 2.10
    # HDF5 data. Keras 3 selects a loader from the suffix, so use a temporary
    # .h5 alias when needed.
    if model_path.suffix.lower() in {".h5", ".hdf5"}:
        model.load_weights(model_path)
        return
    with tempfile.TemporaryDirectory(prefix="crnn_keras_weights_") as temp_dir:
        compatible_path = Path(temp_dir) / "model_weights.h5"
        shutil.copy2(model_path, compatible_path)
        model.load_weights(compatible_path)


def _load_audio_statistics(
    mean_path: str | Path,
    std_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(np.load(Path(mean_path)), dtype=np.float32).reshape(-1)
    std = np.asarray(np.load(Path(std_path)), dtype=np.float32).reshape(-1)
    if mean.shape != (N_MFCC,) or std.shape != (N_MFCC,):
        raise ValueError(
            f"Statistik MFCC harus berbentuk {(N_MFCC,)}, "
            f"ditemukan mean={mean.shape}, std={std.shape}."
        )
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Statistik MFCC berisi NaN atau Inf.")
    if np.any(std <= 0):
        raise ValueError("Standar deviasi MFCC harus lebih besar dari nol.")
    return mean, std


def load_model_bundle(
    model_path: str | Path,
    mean_path: str | Path,
    std_path: str | Path,
    base_modality_scale: float = 1.0,
) -> ModelBundle:
    if not 0.0 < base_modality_scale <= 1.0:
        raise ValueError("base_modality_scale harus lebih dari 0 dan maksimal 1")
    base_model = build_model()
    resolved_model_path = Path(model_path).resolve()
    _load_weights_compatible(base_model, resolved_model_path)

    encoder = models.Model(
        base_model.inputs,
        [
            base_model.get_layer("visual_dropout").output,
            base_model.get_layer("audio_dropout").output,
        ],
        name="modality_encoder",
    )

    visual_features = layers.Input((128,), name="visual_features_input")
    audio_features = layers.Input((128,), name="audio_features_input")
    fusion = base_model.get_layer("model_level_fusion")(
        [visual_features, audio_features]
    )
    fusion = base_model.get_layer("fusion_dense_256")(fusion)
    fusion = base_model.get_layer("fusion_batchnorm")(fusion)
    fusion = base_model.get_layer("fusion_relu_256")(fusion)
    fusion = base_model.get_layer("fusion_dropout_40")(fusion)
    fusion = base_model.get_layer("fusion_dense_128")(fusion)
    fusion = base_model.get_layer("fusion_dropout_30")(fusion)
    output = base_model.get_layer("emotion_output")(fusion)
    fusion_head = models.Model(
        [visual_features, audio_features], output, name="weighted_fusion_head"
    )

    # Trace the entire inference path once on the Streamlit script thread.
    # WebRTC predictions then execute an immutable TensorFlow graph instead of
    # entering Keras Python name scopes from a background worker. This avoids
    # a Keras 3 race where a Streamlit rerun can reset thread-local name-scope
    # state while live inference is still active.
    @tf.function(
        input_signature=[
            tf.TensorSpec(
                shape=(None, NUM_FRAMES, IMAGE_SIZE, IMAGE_SIZE, 3),
                dtype=tf.float32,
                name="visual_input",
            ),
            tf.TensorSpec(
                shape=(None, *AUDIO_SHAPE),
                dtype=tf.float32,
                name="audio_input",
            ),
            tf.TensorSpec(shape=(), dtype=tf.float32, name="visual_scale"),
            tf.TensorSpec(shape=(), dtype=tf.float32, name="audio_scale"),
        ],
        autograph=False,
    )
    def inference_function(
        visual_tensor: tf.Tensor,
        audio_tensor: tf.Tensor,
        visual_scale: tf.Tensor,
        audio_scale: tf.Tensor,
    ) -> tf.Tensor:
        encoded_visual, encoded_audio = encoder(
            {
                "visual_input": visual_tensor,
                "audio_input": audio_tensor,
            },
            training=False,
        )
        return fusion_head(
            [
                encoded_visual * visual_scale,
                encoded_audio * audio_scale,
            ],
            training=False,
        )

    # Force tracing/warm-up before the model is handed to a WebRTC worker.
    inference_function(
        tf.zeros(
            (1, NUM_FRAMES, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=tf.float32
        ),
        tf.zeros((1, *AUDIO_SHAPE), dtype=tf.float32),
        tf.constant(1.0, dtype=tf.float32),
        tf.constant(1.0, dtype=tf.float32),
    )

    train_mean, train_std = _load_audio_statistics(mean_path, std_path)
    return ModelBundle(
        base_model=base_model,
        encoder=encoder,
        fusion_head=fusion_head,
        inference_function=inference_function,
        train_mean=train_mean,
        train_std=train_std,
        model_path=resolved_model_path,
        base_modality_scale=float(base_modality_scale),
        lock=threading.Lock(),
    )


def create_face_detector() -> cv2.CascadeClassifier:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError(f"Haar Cascade gagal dimuat: {cascade_path}")
    return detector


def detect_largest_face(
    frame_bgr: np.ndarray,
    detector: cv2.CascadeClassifier,
    max_detection_side: int | None = None,
) -> tuple[int, int, int, int] | None:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    height, width = equalized.shape[:2]

    scale = 1.0
    if max_detection_side is not None and max(height, width) > max_detection_side:
        scale = max_detection_side / max(height, width)
        equalized = cv2.resize(
            equalized,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    minimum = max(24, round(40 * scale))
    faces = detector.detectMultiScale(
        equalized,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(minimum, minimum),
    )
    if not len(faces):
        return None

    x, y, face_width, face_height = max(
        faces, key=lambda box: int(box[2]) * int(box[3])
    )
    inverse_scale = 1.0 / scale
    return (
        round(x * inverse_scale),
        round(y * inverse_scale),
        round(face_width * inverse_scale),
        round(face_height * inverse_scale),
    )


def _center_crop(frame_rgb: np.ndarray) -> np.ndarray:
    height, width = frame_rgb.shape[:2]
    side = min(height, width)
    y0 = (height - side) // 2
    x0 = (width - side) // 2
    return frame_rgb[y0 : y0 + side, x0 : x0 + side]


def preprocess_face_frame(
    frame_bgr: np.ndarray,
    detector: cv2.CascadeClassifier,
) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    box = detect_largest_face(frame_bgr, detector)
    if box is None:
        face_rgb = _center_crop(frame_rgb)
    else:
        x, y, width, height = box
        margin_x = round(width * 0.10)
        margin_y = round(height * 0.10)
        x0 = max(0, x - margin_x)
        y0 = max(0, y - margin_y)
        x1 = min(frame_rgb.shape[1], x + width + margin_x)
        y1 = min(frame_rgb.shape[0], y + height + margin_y)
        face_rgb = frame_rgb[y0:y1, x0:x1]

    if face_rgb.size == 0:
        raise RuntimeError("Area wajah kosong setelah cropping.")
    resized = cv2.resize(
        face_rgb,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    return resized.astype(np.float32) / 255.0, box


def preprocess_frames(
    frames: Sequence[np.ndarray],
) -> tuple[np.ndarray, dict[str, object]]:
    if not frames:
        raise RuntimeError("Tidak ada frame video yang dapat diproses.")
    detector = create_face_detector()
    indices = np.linspace(0, len(frames) - 1, NUM_FRAMES, dtype=int)
    processed: list[np.ndarray] = []
    detected_count = 0
    representative_box = None
    for index in indices:
        face, box = preprocess_face_frame(frames[int(index)], detector)
        processed.append(face)
        if box is not None:
            detected_count += 1
            representative_box = box

    visual = np.asarray(processed, dtype=np.float32)[np.newaxis, ...]
    expected = (1, NUM_FRAMES, IMAGE_SIZE, IMAGE_SIZE, 3)
    if visual.shape != expected:
        raise RuntimeError(f"Shape visual {visual.shape}, seharusnya {expected}")
    return visual, {
        "face_detected_count": detected_count,
        "center_crop_count": NUM_FRAMES - detected_count,
        "representative_box": representative_box,
    }


def _resample_waveform(
    waveform: np.ndarray,
    source_sample_rate: int,
    target_sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if source_sample_rate == target_sample_rate or not waveform.size:
        return waveform
    divisor = math.gcd(int(source_sample_rate), int(target_sample_rate))
    result = scipy_signal.resample_poly(
        waveform,
        up=target_sample_rate // divisor,
        down=source_sample_rate // divisor,
    )
    return np.asarray(result, dtype=np.float32)


def _hz_to_slaney_mel(frequencies: np.ndarray) -> np.ndarray:
    frequencies = np.asarray(frequencies, dtype=np.float64)
    frequency_spacing = 200.0 / 3.0
    mel = frequencies / frequency_spacing
    minimum_log_hz = 1000.0
    minimum_log_mel = minimum_log_hz / frequency_spacing
    log_step = math.log(6.4) / 27.0
    logarithmic = frequencies >= minimum_log_hz
    mel[logarithmic] = minimum_log_mel + np.log(
        frequencies[logarithmic] / minimum_log_hz
    ) / log_step
    return mel


def _slaney_mel_to_hz(mels: np.ndarray) -> np.ndarray:
    mels = np.asarray(mels, dtype=np.float64)
    frequency_spacing = 200.0 / 3.0
    frequencies = frequency_spacing * mels
    minimum_log_hz = 1000.0
    minimum_log_mel = minimum_log_hz / frequency_spacing
    log_step = math.log(6.4) / 27.0
    logarithmic = mels >= minimum_log_mel
    frequencies[logarithmic] = minimum_log_hz * np.exp(
        log_step * (mels[logarithmic] - minimum_log_mel)
    )
    return frequencies


def _mel_filter_bank() -> np.ndarray:
    mel_min, mel_max = _hz_to_slaney_mel(
        np.asarray([0.0, SAMPLE_RATE / 2.0])
    )
    mel_frequencies = _slaney_mel_to_hz(
        np.linspace(mel_min, mel_max, N_MELS + 2)
    )
    fft_frequencies = np.linspace(0.0, SAMPLE_RATE / 2.0, 1 + N_FFT // 2)
    frequency_differences = np.diff(mel_frequencies)
    ramps = mel_frequencies[:, np.newaxis] - fft_frequencies[np.newaxis, :]
    lower = -ramps[:-2] / frequency_differences[:-1, np.newaxis]
    upper = ramps[2:] / frequency_differences[1:, np.newaxis]
    weights = np.maximum(0.0, np.minimum(lower, upper))
    normalization = 2.0 / (
        mel_frequencies[2 : N_MELS + 2] - mel_frequencies[:N_MELS]
    )
    weights *= normalization[:, np.newaxis]
    return weights.astype(np.float32)


MEL_FILTER_BANK = _mel_filter_bank()


def extract_mfcc(waveform: np.ndarray) -> np.ndarray:
    """Compute the librosa-default MFCC used by the training pipeline."""
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if not waveform.size:
        raise RuntimeError("Audio kosong.")

    padded = np.pad(waveform, N_FFT // 2, mode="constant")
    frame_count = 1 + (padded.size - N_FFT) // HOP_LENGTH
    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[
        : frame_count * HOP_LENGTH : HOP_LENGTH
    ]
    short_window = scipy_signal.get_window(
        "hann", WIN_LENGTH, fftbins=True
    ).astype(np.float32)
    left_padding = (N_FFT - WIN_LENGTH) // 2
    right_padding = N_FFT - WIN_LENGTH - left_padding
    window = np.pad(short_window, (left_padding, right_padding))
    spectrum = np.fft.rfft(frames * window[np.newaxis, :], n=N_FFT, axis=1)
    power = (np.abs(spectrum) ** 2).astype(np.float32).T
    mel_power = np.maximum(MEL_FILTER_BANK @ power, 1e-10)
    mel_db = 10.0 * np.log10(mel_power)
    mel_db = np.maximum(mel_db, float(mel_db.max()) - 80.0)
    coefficients = scipy_fft.dct(mel_db, type=2, axis=0, norm="ortho")
    return np.asarray(coefficients[:N_MFCC], dtype=np.float32)


def preprocess_audio(
    waveform: np.ndarray,
    source_sample_rate: int,
    train_mean: np.ndarray,
    train_std: np.ndarray,
) -> np.ndarray:
    if source_sample_rate <= 0:
        raise ValueError("Sample rate audio harus lebih besar dari nol.")
    waveform = _resample_waveform(waveform, source_sample_rate)
    mfcc = extract_mfcc(waveform)
    normalized = (mfcc - train_mean[:, np.newaxis]) / (
        train_std[:, np.newaxis] + 1e-8
    )
    if normalized.shape[1] < TARGET_TIME_STEPS:
        normalized = np.pad(
            normalized,
            ((0, 0), (0, TARGET_TIME_STEPS - normalized.shape[1])),
        )
    else:
        normalized = normalized[:, :TARGET_TIME_STEPS]
    result = normalized[..., np.newaxis].astype(np.float32)[np.newaxis, ...]
    expected = (1, *AUDIO_SHAPE)
    if result.shape != expected:
        raise RuntimeError(f"Shape audio {result.shape}, seharusnya {expected}")
    return result


def empty_audio_input() -> np.ndarray:
    return np.zeros((1, *AUDIO_SHAPE), dtype=np.float32)


def audio_frame_to_mono(frame: object) -> tuple[np.ndarray, int]:
    array = np.asarray(frame.to_ndarray())
    sample_rate = int(getattr(frame, "sample_rate", 0) or 48_000)
    frame_format = getattr(frame, "format", None)
    frame_layout = getattr(frame, "layout", None)
    is_planar = bool(getattr(frame_format, "is_planar", False))
    channels = getattr(frame_layout, "channels", ())
    channel_count = max(1, len(channels))

    if array.ndim == 2:
        if is_planar:
            array = array.mean(axis=0)
        elif array.shape[0] == 1 and channel_count > 1:
            packed = array.reshape(-1)
            usable = packed.size - (packed.size % channel_count)
            array = packed[:usable].reshape(-1, channel_count).mean(axis=1)
        elif array.shape[-1] == channel_count:
            array = array.mean(axis=-1)
        else:
            array = array.reshape(-1)

    array = array.reshape(-1)
    if np.issubdtype(array.dtype, np.integer):
        maximum = float(max(abs(np.iinfo(array.dtype).min), np.iinfo(array.dtype).max))
        array = array.astype(np.float32) / maximum
    else:
        array = array.astype(np.float32)
    return np.clip(array, -1.0, 1.0), sample_rate


def merge_audio_chunks(
    chunks: Sequence[tuple[np.ndarray, int]],
) -> np.ndarray:
    # Resampling every 10-20 ms WebRTC packet separately creates filter-edge
    # artifacts. Concatenate consecutive packets with the same rate first,
    # then resample each continuous run only once.
    resampled_runs: list[np.ndarray] = []
    current_rate: int | None = None
    current_chunks: list[np.ndarray] = []

    def flush_run() -> None:
        if current_rate is None or not current_chunks:
            return
        continuous = np.concatenate(current_chunks).astype(np.float32, copy=False)
        resampled_runs.append(_resample_waveform(continuous, current_rate))

    for waveform, sample_rate in chunks:
        waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if not waveform.size:
            continue
        sample_rate = int(sample_rate)
        if current_rate is None:
            current_rate = sample_rate
        elif sample_rate != current_rate:
            flush_run()
            current_chunks = []
            current_rate = sample_rate
        current_chunks.append(waveform)
    flush_run()

    if not resampled_runs:
        return np.asarray([], dtype=np.float32)
    return np.concatenate(resampled_runs).astype(np.float32, copy=False)


def measure_audio_quality(
    waveform: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> AudioQuality:
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if not waveform.size:
        return AudioQuality(
            rms=0.0,
            peak=0.0,
            dbfs=-120.0,
            duration_seconds=0.0,
            has_usable_signal=False,
        )
    centered = waveform - float(np.mean(waveform, dtype=np.float64))
    rms = float(np.sqrt(np.mean(np.square(centered), dtype=np.float64)))
    peak = float(np.max(np.abs(centered)))
    dbfs = float(20.0 * np.log10(max(rms, 1e-6)))
    duration = waveform.size / max(1, int(sample_rate))
    # The quietest training utterances are around RMS 0.0009 and peak 0.0069.
    # Values below both limits are normally silence or an inactive microphone.
    has_usable_signal = rms >= 0.0010 or peak >= 0.0060
    return AudioQuality(
        rms=rms,
        peak=peak,
        dbfs=dbfs,
        duration_seconds=duration,
        has_usable_signal=has_usable_signal,
    )


def annotate_frame(
    frame_bgr: np.ndarray,
    prediction: Prediction | None,
    detector: cv2.CascadeClassifier | None = None,
    status_text: str | None = None,
    face_box: tuple[int, int, int, int] | None = None,
    detect_face: bool = True,
    confidence_threshold: float = 0.0,
) -> tuple[np.ndarray, bool]:
    result = frame_bgr.copy()
    if detect_face:
        detector = detector or create_face_detector()
        face_box = detect_largest_face(
            result, detector, max_detection_side=480
        )
    face_detected = face_box is not None

    if prediction is not None:
        if prediction.confidence < confidence_threshold:
            color = (0, 165, 255)
            label_text = f"TIDAK YAKIN  {prediction.confidence * 100:.1f}%"
        else:
            color = CLASS_COLORS[prediction.label]
            label_text = (
                f"{prediction.display_label.upper()}  "
                f"{prediction.confidence * 100:.1f}%"
            )
    else:
        color = (0, 210, 255)
        label_text = status_text or "Mengumpulkan data..."

    if face_box is not None:
        x, y, width, height = face_box
        cv2.rectangle(result, (x, y), (x + width, y + height), color, 3)
        cv2.putText(
            result,
            label_text,
            (x, max(30, y - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            color,
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            result,
            f"Wajah belum ditemukan | {label_text}",
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
    return result, face_detected


def sample_video_frames(
    capture: cv2.VideoCapture,
    start_seconds: float,
    end_seconds: float,
) -> list[np.ndarray]:
    if end_seconds <= start_seconds:
        return []
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 25.0
    frame_span = max(1, int(math.ceil((end_seconds - start_seconds) * fps)))
    target_indices = np.linspace(0, frame_span - 1, NUM_FRAMES, dtype=int)
    target_counts = np.bincount(target_indices, minlength=frame_span)
    frames: list[np.ndarray] = []
    capture.set(cv2.CAP_PROP_POS_MSEC, float(start_seconds) * 1000.0)
    for frame_index in range(frame_span):
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        repeat = int(target_counts[frame_index])
        if repeat:
            frames.extend(frame.copy() for _ in range(repeat))
    return frames


def _extract_audio(video_path: Path, wav_path: Path) -> tuple[bool, str | None]:
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
        str(SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        creationflags=(
            subprocess.CREATE_NO_WINDOW
            if hasattr(subprocess, "CREATE_NO_WINDOW")
            else 0
        ),
    )
    if completed.returncode == 0 and wav_path.is_file():
        return True, None
    message = completed.stderr.strip() or "FFmpeg tidak menemukan track audio."
    return False, message


def analyze_video(
    video_path: str | Path,
    bundle: ModelBundle,
    audio_focus: float,
    segment_seconds: float = DEFAULT_WINDOW_SECONDS,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[list[SegmentPrediction], list[str]]:
    if not 1.0 <= segment_seconds <= 5.0:
        raise ValueError("Durasi segmen harus berada pada rentang 1 sampai 5 detik.")
    path = Path(video_path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Video tidak dapat dibuka: {path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or frame_count <= 0:
        capture.release()
        raise RuntimeError("Metadata durasi video tidak valid.")
    duration = frame_count / fps
    full_segments = int(duration // segment_seconds)
    remainder = duration - full_segments * segment_seconds
    if full_segments == 0:
        segment_count = 1
    elif 0.05 < remainder < MIN_RELIABLE_SEGMENT_SECONDS:
        # Absorb a very short tail into the previous segment instead of making
        # a prediction from a heavily padded audio window.
        segment_count = full_segments
    else:
        segment_count = full_segments + int(remainder > 0.05)
    warnings: list[str] = []
    if duration < MIN_RELIABLE_SEGMENT_SECONDS:
        warnings.append(
            f"Video hanya berdurasi {duration:.1f} detik; input audio "
            "dipanjangkan dengan padding seperti pada preprocessing training."
        )
    elif (
        0.05 < remainder < MIN_RELIABLE_SEGMENT_SECONDS
        and full_segments > 0
    ):
        warnings.append(
            f"Sisa video {remainder:.1f} detik digabung ke segmen sebelumnya "
            "agar tidak menghasilkan prediksi dari potongan yang terlalu pendek."
        )
    elif (
        remainder >= MIN_RELIABLE_SEGMENT_SECONDS
        and remainder < segment_seconds - 0.05
    ):
        warnings.append(
            f"Segmen terakhir berdurasi {remainder:.1f} detik; input audio "
            "dipanjangkan dengan padding seperti pada preprocessing training."
        )
    results: list[SegmentPrediction] = []

    with tempfile.TemporaryDirectory(prefix="crnn_video_audio_") as temp_dir:
        wav_path = Path(temp_dir) / "audio.wav"
        has_audio, audio_error = _extract_audio(path, wav_path)

        audio_file = None
        if has_audio:
            try:
                audio_file = sf.SoundFile(wav_path)
            except Exception:
                has_audio = False

        if not has_audio:
            warnings.append(
                "Video tidak memiliki track audio yang dapat dibaca. "
                "Prediksi tetap dijalankan menggunakan fitur visual."
            )

        silent_segment_count = 0

        try:
            for segment_index in range(segment_count):
                start = segment_index * segment_seconds
                end = (
                    duration
                    if segment_index == segment_count - 1
                    else min(duration, start + segment_seconds)
                )
                frames = sample_video_frames(capture, start, end)
                if not frames:
                    warnings.append(
                        f"Segmen {start:.1f}-{end:.1f} detik dilewati karena "
                        "frame gagal dibaca."
                    )
                    if progress_callback:
                        progress_callback(segment_index + 1, segment_count)
                    continue

                visual, visual_info = preprocess_frames(frames)

                # Nilai bawaan untuk video tanpa audio atau segmen terlalu hening.
                audio = empty_audio_input()
                effective_audio_focus = 0.0

                if audio_file is not None:
                    audio_file.seek(int(start * audio_file.samplerate))
                    requested_frames = max(
                        1, int(round((end - start) * audio_file.samplerate))
                    )
                    waveform = audio_file.read(
                        frames=requested_frames,
                        dtype="float32",
                        always_2d=False,
                    )
                    waveform_array = np.asarray(
                        waveform, dtype=np.float32
                    ).reshape(-1)
                    audio_quality = measure_audio_quality(
                        waveform_array,
                        int(audio_file.samplerate),
                    )

                    if waveform_array.size and audio_quality.has_usable_signal:
                        audio = preprocess_audio(
                            waveform_array,
                            int(audio_file.samplerate),
                            bundle.train_mean,
                            bundle.train_std,
                        )
                        effective_audio_focus = audio_focus
                    else:
                        silent_segment_count += 1

                prediction = bundle.predict(
                    visual,
                    audio,
                    effective_audio_focus,
                )
                preview = frames[len(frames) // 2]
                annotated, overlay_face = annotate_frame(preview, prediction)
                results.append(
                    SegmentPrediction(
                        start_seconds=start,
                        end_seconds=end,
                        prediction=prediction,
                        face_detected=(
                            overlay_face
                            or int(visual_info["face_detected_count"]) > 0
                        ),
                        preview_bgr=annotated,
                    )
                )
                if progress_callback:
                    progress_callback(segment_index + 1, segment_count)
        finally:
            if audio_file is not None:
                audio_file.close()
            capture.release()

        if silent_segment_count:
            warnings.append(
                f"{silent_segment_count} segmen memiliki audio terlalu hening "
                "dan dianalisis menggunakan fitur visual."
            )

    return results, warnings


def self_test(bundle: ModelBundle) -> dict[str, object]:
    visual = np.full(
        (1, NUM_FRAMES, IMAGE_SIZE, IMAGE_SIZE, 3),
        0.5,
        dtype=np.float32,
    )
    audio = empty_audio_input()
    predictions = {
        focus: bundle.predict(visual, audio, audio_focus=focus)
        for focus in (0.0, 0.5, 1.0)
    }
    for prediction in predictions.values():
        if not np.isfinite(prediction.probabilities).all():
            raise RuntimeError("Self-test menghasilkan NaN atau Inf.")
        if not np.isclose(prediction.probabilities.sum(), 1.0, atol=1e-5):
            raise RuntimeError("Probabilitas self-test tidak berjumlah satu.")
    midpoint = predictions[0.5]
    return {
        "model": str(bundle.model_path),
        "base_modality_scale": bundle.base_modality_scale,
        "visual_shape": visual.shape,
        "audio_shape": audio.shape,
        "output_shape": midpoint.probabilities.shape,
        "probability_sum": float(midpoint.probabilities.sum()),
        "prediction": midpoint.display_label,
        "focus_modes": {
            str(focus): item.display_label for focus, item in predictions.items()
        },
    }
