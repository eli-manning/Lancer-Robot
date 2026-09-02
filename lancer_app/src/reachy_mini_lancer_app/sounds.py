"""
Non-speech audio cues.

Lancer goes quiet while the server transcribes, retrieves, calls a skill and
generates. Even at a second or two that silence reads as "it didn't hear me",
and people repeat themselves into the gap -- which the microphone then captures
as a second question. A quiet cue is the cheapest fix: it says "still here,
working" without claiming the floor.

Everything is generated with numpy rather than shipped as WAV files, so the app
stays installable with no binary assets and the cues can be retuned by editing
numbers instead of re-recording.

Volume is the important setting. These play through the same speaker the
answers do, and the mic hears them: too loud and the barge-in detector treats
Lancer's own thinking noise as someone interrupting. The default is deliberately
below the echo floor the barge-in gate calibrates against.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

# A soft two-tone pulse, roughly a heartbeat's pace. Sine waves with an
# envelope, not square waves: a hard edge sounds like an error tone, and people
# read error tones as "it broke".
_PULSE_HZ = (392.0, 523.25)     # G4, C5 -- a fifth apart, so it reads as neutral
_PULSE_LEN = 0.11               # seconds per tone
_PULSE_GAP = 0.62               # seconds of silence between pulses


def _tone(frequency: float, seconds: float, volume: float,
          fade: float = 0.02) -> np.ndarray:
    """One sine tone with fade in/out, so it does not click at the edges."""
    samples = int(SAMPLE_RATE * seconds)
    t = np.arange(samples, dtype=np.float32) / SAMPLE_RATE
    wave = np.sin(2 * np.pi * frequency * t).astype(np.float32) * volume

    edge = max(1, int(SAMPLE_RATE * fade))
    if samples > 2 * edge:
        ramp = np.linspace(0.0, 1.0, edge, dtype=np.float32)
        wave[:edge] *= ramp
        wave[-edge:] *= ramp[::-1]
    return wave


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)


def thinking_pulse(volume: float) -> np.ndarray:
    """One cycle of the working cue: two soft tones, then a gap."""
    return np.concatenate([
        _tone(_PULSE_HZ[0], _PULSE_LEN, volume),
        _tone(_PULSE_HZ[1], _PULSE_LEN, volume * 0.8),
        _silence(_PULSE_GAP),
    ])


def skill_chirp(volume: float) -> np.ndarray:
    """Rising pair: Lancer is going out to fetch something."""
    return np.concatenate([
        _tone(523.25, 0.07, volume * 1.3),
        _tone(783.99, 0.09, volume * 1.3),
    ])


def error_blip(volume: float) -> np.ndarray:
    """Falling pair: that lookup did not work."""
    return np.concatenate([
        _tone(440.0, 0.09, volume * 1.2),
        _tone(311.13, 0.13, volume * 1.2),
    ])


def heard_blip(volume: float) -> np.ndarray:
    """A single short tick acknowledging that speech was captured."""
    return _tone(659.25, 0.06, volume * 1.1)


class Cues:
    """
    Plays cues on the robot's speaker without blocking the caller.

    One thread, one flag. start_thinking() loops the pulse until stop() is
    called; the one-shot cues push and return. Every push is wrapped: a media
    pipeline that is not ready is a reason to skip a sound effect, never a
    reason to fail a conversation.
    """

    def __init__(self, mini, volume: float = 0.06, enabled: bool = True):
        self.mini = mini
        self.volume = float(volume)
        self.enabled = bool(enabled)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _push(self, mono: np.ndarray) -> None:
        try:
            # The output pipeline is 2-channel; mono is read as interleaved
            # stereo and plays at double speed.
            stereo = np.column_stack([mono, mono]).astype(np.float32)
            self.mini.media.push_audio_sample(stereo)
        except Exception as e:
            logger.debug("Cue playback skipped: %s", e)

    def play(self, cue: np.ndarray) -> None:
        if self.enabled and self.volume > 0:
            self._push(cue)

    def skill(self) -> None:
        self.play(skill_chirp(self.volume))

    def error(self) -> None:
        self.play(error_blip(self.volume))

    def heard(self) -> None:
        self.play(heard_blip(self.volume))

    def start_thinking(self) -> None:
        """Loop the working cue until stop_thinking()."""
        if not self.enabled or self.volume <= 0 or self._thread:
            return
        self._stop.clear()

        def loop() -> None:
            pulse = thinking_pulse(self.volume)
            period = len(pulse) / SAMPLE_RATE
            while not self._stop.is_set():
                self._push(pulse)
                # Sleep the pulse's own length so the queue does not build up a
                # backlog that keeps playing after the answer has started.
                if self._stop.wait(period):
                    break

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop_thinking(self) -> None:
        """Stop the loop and drop whatever is still queued."""
        if not self._thread:
            return
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._thread = None
        # push_audio_sample queues; without clearing it the last pulse plays
        # over the first words of the answer. Restarting the pipeline is what
        # actually drops queued audio -- the same trick barge-in uses.
        try:
            self.mini.media.stop_playing()
            self.mini.media.start_playing()
        except Exception as e:
            logger.debug("Could not flush cue audio: %s", e)
        # Let the speaker settle before the answer is pushed, or the restart
        # clips the first syllable.
        time.sleep(0.05)
