"""
Stage 2 - Feature extraction for the digit model. Identical approach to
src/features.py (log-mel spectrogram, soundfile-based loading to avoid
the optional torchcodec dependency some torchaudio versions require).
"""
import torch
import torchaudio
import soundfile as sf

from . import config

_mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=config.SAMPLE_RATE,
    n_fft=config.N_FFT,
    hop_length=config.HOP_LENGTH,
    n_mels=config.N_MELS,
)
_db_transform = torchaudio.transforms.AmplitudeToDB()


def load_and_fix_length(path, target_samples=config.CLIP_SAMPLES, sr=config.SAMPLE_RATE):
    data, orig_sr = sf.read(str(path), dtype="float32", always_2d=True)  # (samples, channels)
    waveform = torch.from_numpy(data.T)  # (channels, samples)

    if waveform.shape[0] > 1:  # stereo -> mono
        waveform = waveform.mean(dim=0, keepdim=True)
    if orig_sr != sr:
        waveform = torchaudio.functional.resample(waveform, orig_sr, sr)

    length = waveform.shape[1]
    if length < target_samples:
        pad = target_samples - length
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    elif length > target_samples:
        waveform = waveform[:, :target_samples]
    return waveform


def waveform_to_logmel(waveform: torch.Tensor) -> torch.Tensor:
    mel = _mel_transform(waveform)
    logmel = _db_transform(mel)
    return logmel


def path_to_features(path) -> torch.Tensor:
    waveform = load_and_fix_length(path)
    return waveform_to_logmel(waveform)
