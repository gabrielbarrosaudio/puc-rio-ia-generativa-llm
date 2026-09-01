"""
Stage 5/6 - Real-time inference + thread-safe bridge to the game.

VoiceCommandListener runs in a background thread:
  1. Continuously records audio into a rolling 1-second buffer via
     sounddevice (this runs on sounddevice's own internal audio thread).
  2. On a separate polling thread, uses simple energy-based VAD (root-mean-
     -square amplitude) to decide "is someone probably talking right now?"
     -- this avoids running the (relatively expensive) model on pure
     silence/background noise on every single poll.
  3. When triggered, runs the last ~1s of audio through the trained model.
  4. If the softmax confidence clears CONFIDENCE_THRESHOLD, the predicted
     label is NOT "silence"/"unknown", and enough time has passed since the
     last accepted command (COOLDOWN_SECONDS, to avoid one spoken word
     firing multiple times), it pushes the label onto a thread-safe queue.

The Pong game (running in the MAIN thread, inside its own while-loop) polls
that queue non-blockingly on every frame with get_command(). This keeps
Turtle's event loop and the audio thread fully decoupled -- neither blocks
the other, which matters because turtle.Screen.update() and sounddevice's
callback both expect to run without being stalled by slow work elsewhere.
"""
import queue
import threading
import time

import numpy as np
import sounddevice as sd
import torch

from . import config
from .features import waveform_to_logmel
from .model import VoiceCommandCNN


class VoiceCommandListener:
    def __init__(self, checkpoint_path=None, energy_threshold=0.01):
        self.device = torch.device("cpu")
        self.model = VoiceCommandCNN().to(self.device)
        ckpt = checkpoint_path or (config.CHECKPOINT_DIR / "best_model.pt")
        if not ckpt.exists():
            raise FileNotFoundError(
                f"No checkpoint at {ckpt}. Run train.py (Stage 4) first."
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

    # --- audio callback: runs on sounddevice's internal audio thread ---
    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"[audio warning] {status}")
        with self._lock:
            self._buffer = np.roll(self._buffer, -frames)
            self._buffer[-frames:] = indata[:, 0]

    # --- inference loop: runs on our own polling thread ---
    def _infer_loop(self, poll_interval=0.2):
        while self._running:
            time.sleep(poll_interval)
            with self._lock:
                snippet = self._buffer.copy()

            rms = float(np.sqrt(np.mean(snippet ** 2)))
            if rms < self.energy_threshold:
                continue  # likely silence; skip inference to save CPU

            waveform = torch.from_numpy(snippet).unsqueeze(0)  # (1, samples)
            features = waveform_to_logmel(waveform).unsqueeze(0)  # (1,1,n_mels,frames)

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
        """Non-blocking. Returns the next predicted label (str), or None."""
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None


if __name__ == "__main__":
    # Quick manual test: prints recognized commands to the terminal without
    # needing the game at all. Useful for tuning CONFIDENCE_THRESHOLD /
    # COOLDOWN_SECONDS / energy_threshold in config.py before wiring up Pong.
    listener = VoiceCommandListener()
    listener.start()
    print("Listening... speak a command (Ctrl+C to stop)")
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
