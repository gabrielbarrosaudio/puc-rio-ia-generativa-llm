"""
Stage 5 - Real-time inference for the digit model. Same design as
src/bridge.py (rolling buffer + energy-based VAD + confidence threshold +
cooldown debounce, thread-safe queue) -- see that file for the full
rationale. Kept as a separate class/module because it wraps a completely
different model/checkpoint/label set.
"""
import queue
import threading
import time

import numpy as np
import sounddevice as sd
import torch

from . import config
from .features import waveform_to_logmel
from .model import DigitCommandCNN


class VoiceCommandListener:
    def __init__(self, checkpoint_path=None, energy_threshold=0.01):
        self.device = torch.device("cpu")
        self.model = DigitCommandCNN().to(self.device)
        ckpt = checkpoint_path or (config.CHECKPOINT_DIR / "best_model.pt")
        if not ckpt.exists():
            raise FileNotFoundError(
                f"No checkpoint at {ckpt}. Run digits.train first."
            )
        self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
        self.model.eval()

        self.energy_threshold = energy_threshold
        self.command_queue = queue.Queue()
        self._buffer = np.zeros(config.CLIP_SAMPLES, dtype=np.float32)
        self._lock = threading.Lock()
        self._last_accept_time = 0.0
        self._running = False
        self._stream = None
        self._thread = None

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"[digits audio warning] {status}")
        with self._lock:
            self._buffer = np.roll(self._buffer, -frames)
            self._buffer[-frames:] = indata[:, 0]

    def _infer_loop(self, poll_interval=0.2):
        while self._running:
            time.sleep(poll_interval)
            with self._lock:
                snippet = self._buffer.copy()

            rms = float(np.sqrt(np.mean(snippet ** 2)))
            if rms < self.energy_threshold:
                continue

            waveform = torch.from_numpy(snippet).unsqueeze(0)
            features = waveform_to_logmel(waveform).unsqueeze(0)

            with torch.no_grad():
                logits = self.model(features)
                probs = torch.softmax(logits, dim=1).squeeze(0)
                conf, pred_idx = torch.max(probs, dim=0)

            label = config.IDX_TO_LABEL[pred_idx.item()]
            now = time.time()
            if (label not in ("silence", "unknown")
                    and conf.item() >= config.CONFIDENCE_THRESHOLD
                    and (now - self._last_accept_time) >= config.COOLDOWN_SECONDS):
                self._last_accept_time = now
                self.command_queue.put(label)

    def start(self):
        self._running = True
        self._stream = sd.InputStream(
            samplerate=config.SAMPLE_RATE, channels=1,
            callback=self._audio_callback,
        )
        self._stream.start()
        self._thread = threading.Thread(target=self._infer_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def get_command(self):
        """Non-blocking. Returns the next recognized word ('one'..'six'), or None."""
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None


if __name__ == "__main__":
    listener = VoiceCommandListener()
    listener.start()
    print("Listening for numbers (one-six)... Ctrl+C to stop")
    try:
        while True:
            cmd = listener.get_command()
            if cmd:
                print(f"-> recognized: {cmd}")
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()