"""
Lancer -- CBU's RAG-powered campus robot.

Runs on the Reachy Mini and follows the same lifecycle as the stock apps: a
ReachyMiniApp subclass whose run() is driven by the daemon and exits when
stop_event is set.

The robot has no TTS engine and limited RAM, so it only captures and plays
audio. Speech-to-text, retrieval, generation and speech synthesis all happen on
the RAG server (see server.py), reached over the wired link.

    idle      -- native head tracking follows whoever is in frame
    listening -- energy-gated capture from the ReSpeaker array
    thinking  -- head tilts while the server runs RAG
    answering -- head returns to neutral, answer plays through the speaker
    confused  -- antennas droop and head dips when retrieval found nothing
"""

import io
import json
import logging
import math
import os
import threading
import time
import wave

import numpy as np
import requests
from reachy_mini import ReachyMini, ReachyMiniApp
from reachy_mini.utils import create_head_pose

try:
    from . import sounds
except ImportError:          # running this file directly rather than as a package
    import sounds

logger = logging.getLogger(__name__)

# --- Config -----------------------------------------------------------------
# The Mac hosting Chroma + Ollama, reachable over the USB-Ethernet link.
# The wired link is fastest but is not always plugged in; Tailscale always
# works. Try each in order at startup and use whichever answers.
SERVER_CANDIDATES = [
    os.environ.get("LANCER_SERVER"),
    "http://10.5.1.82:7870",        # USB-Ethernet link, ~14 ms
    "http://100.87.191.40:7870",    # Tailscale, ~50 ms
]
SERVER_URL = SERVER_CANDIDATES[1]


def pick_server() -> str:
    """Return the first reachable RAG server, preferring the wired link."""
    global SERVER_URL
    for url in SERVER_CANDIDATES:
        if not url:
            continue
        try:
            if requests.get(f"{url}/health", timeout=4).ok:
                SERVER_URL = url
                logger.info("Using RAG server at %s", url)
                return url
        except Exception:
            continue
    logger.warning("No RAG server reachable; defaulting to %s", SERVER_URL)
    return SERVER_URL
REQUEST_TIMEOUT = 120        # seconds; local LLM generation can be slow

SAMPLE_RATE = 16000          # matches reachy_mini.media AudioBase.SAMPLE_RATE

# --- Voice activity detection ---
# The ReSpeaker applies its own gain and noise suppression, so a simple RMS
# gate is enough here -- webrtcvad would mean adding a compiled dependency.
_NOISE_SETTLE = 1.5          # seconds to let wake-up motor noise die down first
_NOISE_CALIBRATION = 2.0     # seconds of ambient sampled to set the gate
_SPEECH_MULTIPLIER = 2.0     # speech must exceed noise floor by this factor
_SPEECH_RMS_FLOOR = 0.025    # never gate below this, however quiet the room
_SPEECH_RMS_CEILING = 0.045  # never gate above this, or quiet speech is missed
_SILENCE_TIMEOUT = 0.7       # seconds of silence that end an utterance
_MIN_UTTERANCE = 0.6         # ignore anything shorter (coughs, door clicks)
_MIN_VOICED = 0.45           # seconds of actually-loud audio required
_MAX_UTTERANCE = 15.0        # hard cap so a noisy room can't record forever

# --- Barge-in ---
# The board runs software AEC, so the mic feed has most of Lancer's own voice
# removed -- but not all of it, so the barge-in gate sits above the speech gate.
_BARGE_MULTIPLIER = 1.6      # barge-in floor relative to the speech gate
_BARGE_CHUNKS = 5            # consecutive loud reads before we accept an interrupt
_BARGE_GRACE = 0.8           # seconds of playback before barge-in can trigger
_POST_SPEECH_DRAIN = 0.6     # seconds of mic input discarded after Lancer speaks
_BARGE_ECHO_HEADROOM = 1.4   # interrupt must exceed measured echo by this factor

# --- Direction of arrival ---
_DOA_INTERVAL = 0.05         # seconds between DoA polls (ReSpeaker updates ~20Hz)
_DOA_DEADBAND = 0.28         # radians (~16 deg) of change before the body moves
_DOA_MIN_INTERVAL = 1.5      # seconds between body turns, so it cannot jitter
_DOA_SUSTAIN = 2             # consecutive speech reads before turning
_DOA_REACQUIRE = 1.2         # seconds to find someone before turning again
_DOA_RETRY_INTERVAL = 0.6    # shortened gap between turns while re-acquiring
_DOA_HOLD = 2.0              # seconds automatic body yaw stays off after a turn
_FACE_STICKY = 6.0           # seconds a face still counts as present after loss

# --- Hybrid vision ---
# YOLO on the Mac sees whole people, so it holds on to someone who is too close,
# turned away, or off to the side -- all cases where the on-board face detector
# gives up. YuNet stays as the fallback when the Mac is unreachable.
_TRACK_INTERVAL = 0.12       # seconds between frames sent for detection
_TRACK_FALLBACK_AFTER = 2.0  # seconds without a YOLO hit before handing back
_FACE_PRIMARY_GRACE = 0.8    # seconds a face must be gone before YOLO takes over
_YOLO_IDLE_INTERVAL = 0.4    # slower polling while YuNet is doing the work
_TRACK_AIM_DURATION = 0.5    # seconds for each look_at_image move
# YOLO re-detects every frame and the box drifts a few pixels each time. Aiming
# at the raw centre makes the head chase that noise, so smooth it and ignore
# movement too small to be a real person moving.
_AIM_SMOOTHING = 0.45        # EMA weight for each new detection
_AIM_DEADBAND_FRAC = 0.022   # of frame width -- below this, do not move
_AIM_MIN_INTERVAL = 0.18     # seconds between commanded moves
_AIM_RESET_AFTER = 1.5       # seconds of no detection before the filter resets
_DOA_MAX_YAW = 2.0           # radians (~115 deg); beyond this the IK refuses
# How much of a turn toward a voice the head takes before the body is asked for
# any of it.
#
# Zero by default, and that is not timidity. The daemon's head tracker owns the
# head and rewrites its pose several times a second, so a head angle commanded
# here survives for about one frame before being pulled back to whatever the
# face detector last saw -- and with the face detector finding almost nobody,
# that is centre. Giving the head a share therefore does not make it lead the
# turn, it just takes that share away from the body, which is the one actuator
# whose position actually persists. Setting this above zero measurably stopped
# Lancer turning toward speech at all.
#
# Making the head genuinely lead needs the tracker suspended for the duration
# of the turn, which is a larger change than a constant.
_DOA_HEAD_SHARE = 0.0        # radians

# --- Body assist ---
# Automatic body yaw only involves the body past 65 degrees of head rotation
# (inverse_kinematics_safe's max_relative_yaw), which is why Lancer reads as
# head-only for anyone standing in front of it. These bring the body in far
# earlier, without touching the mechanical safety that limit provides.
_BODY_ASSIST_DEADBAND = 0.20   # radians (~11 deg) of head yaw before the body helps
_BODY_ASSIST_EXIT_RATIO = 0.35  # keep going until well inside the deadband
_BODY_ASSIST_GAIN = 0.22       # fraction of the remaining head yaw per step
_BODY_ASSIST_MAX_STEP = 0.05   # radians per step, so ~19 deg/s at this interval
_BODY_ASSIST_INTERVAL = 0.15   # seconds between steps; small and often is smooth
_BODY_ASSIST_HOLD = 1.2        # seconds automatic yaw stays off after a DoA turn
# Self-limiting: a controller that hunts, or one whose corrections never reduce
# the head's offset, shuts itself off and gives the body back.
_BODY_ASSIST_MAX_REVERSALS = 5
_BODY_ASSIST_MAX_STUCK = 25
_DOA_FACE_GRACE = 2.5        # seconds without a face before sound may steer us

# --- Poses ---
_ANTENNA_DROOP = 0.4         # radians, symmetric droop for the confused pose
_ANTENNA_PERK = 0.3          # radians, alert tilt while thinking
# Vertical (0.0) is an unstable equilibrium for the antenna gearboxes -- they
# hunt around it and visibly shake. Pollen's guidance is to bias a few degrees
# off vertical so gravity takes up the backlash.
_ANTENNA_NEUTRAL = 0.18      # radians, ~10 degrees

GREETING = "Hey! I'm Lancer, CBU's ACM robot. Ask me about the university."


class Lancer:
    """Robot behaviour for one Lancer session."""

    def __init__(self, mini: ReachyMini, cfg: dict | None = None):
        self.mini = mini
        self.cfg = cfg or {}
        self.speech_rms = _SPEECH_RMS_FLOOR
        self.speaking = False
        self._last_yaw = 0.0
        self._last_turn = 0.0
        self._last_face = 0.0
        self._last_person = 0.0
        self.vision_mode = "yunet"
        self.request_aim_reset = False
        self._auto_yaw_resume_at = 0.0
        # Measured once per turn from the AEC residual, then reused for every
        # sentence in that turn.
        self._barge_threshold = 0.0
        # Automatic body yaw solves for the body on every head move, and when
        # the head can reach a target on its own it solves for a body yaw of
        # zero -- so it does not merely decline to help, it actively unwinds any
        # turn we make. Off by default; body_assist_loop owns the body instead.
        self._auto_yaw = bool(self.opt("auto_body_yaw", True))
        self._last_body_move = 0.0
        # Set before the robot is put to sleep. The tracking loops each sleep
        # for a fraction of a second between checks, so without this they get
        # one more move in after goto_sleep() has started and fight it on the
        # way down -- which looks like the head twitching as it folds up.
        self.stopping = False
        self.cues = sounds.Cues(
            mini,
            volume=float(self.opt("thinking_sound_volume", 0.06)),
            enabled=bool(self.opt("thinking_sound", True)),
        )

    @property
    def barge_rms(self) -> float:
        return self.speech_rms * _BARGE_MULTIPLIER

    def opt(self, key: str, default):
        value = self.cfg.get(key, default)
        return default if value is None else value

    # -- capture setup -------------------------------------------------------

    @staticmethod
    def _mono(chunk: np.ndarray) -> np.ndarray:
        """The ReSpeaker returns 2 channels; average them to mono."""
        return chunk.mean(axis=1) if chunk.ndim == 2 else chunk

    def calibrate(self) -> None:
        """
        Measure the ambient noise floor and set the speech gate above it.

        The room's baseline varies a lot between a quiet office and a hallway,
        and a fixed threshold either misses quiet speech or triggers on HVAC.
        """
        # Motors settle after wake_up(); sampling too early reads their noise
        # as the room's floor and leaves the gate far too high.
        time.sleep(_NOISE_SETTLE)

        levels: list[float] = []
        deadline = time.monotonic() + _NOISE_CALIBRATION
        while time.monotonic() < deadline:
            chunk = self.mini.media.get_audio_sample()
            if chunk is None or len(chunk) == 0:
                time.sleep(0.01)
                continue
            mono = self._mono(chunk)
            levels.append(float(np.sqrt(np.mean(np.square(mono)))))

        if levels:
            # 25th percentile tracks the true floor even if someone talks or a
            # door slams during calibration; median would be dragged upward.
            noise = float(np.percentile(levels, 25))
            self.speech_rms = min(
                max(noise * float(self.opt("speech_multiplier", _SPEECH_MULTIPLIER)),
                    _SPEECH_RMS_FLOOR),
                float(self.opt("speech_rms_ceiling", _SPEECH_RMS_CEILING)),
            )
        logger.info(
            "Noise floor calibrated: gate=%.4f barge=%.4f (%d samples, p25=%.4f)",
            self.speech_rms, self.barge_rms, len(levels),
            float(np.percentile(levels, 25)) if levels else -1.0,
        )

    # -- gestures ------------------------------------------------------------

    # Expression is antenna-only: head poses fight the daemon's visual tracking,
    # and staying locked on the person matters more than a tilt gesture.

    def thinking(self) -> None:
        self.mini.set_target_antenna_joint_positions([_ANTENNA_PERK, -_ANTENNA_PERK])

    def answering(self) -> None:
        # Mirrored so the pair looks symmetric while both stay off vertical.
        self.mini.set_target_antenna_joint_positions(
            [_ANTENNA_NEUTRAL, -_ANTENNA_NEUTRAL]
        )

    def confused(self) -> None:
        self.mini.set_target_antenna_joint_positions([-_ANTENNA_DROOP, _ANTENNA_DROOP])

    # -- audio ---------------------------------------------------------------

    def listen(self, stop_event: threading.Event,
               max_wait: float | None = None) -> np.ndarray | None:
        """
        Capture one utterance from the mic array.

        Returns float32 samples at SAMPLE_RATE, or None if the app was asked to
        stop or nothing worth sending was captured.

        max_wait bounds how long to wait for speech to *start*. Without it this
        blocks forever, which is right in the idle loop and wrong after a
        barge-in: if the thing that interrupted Lancer was a door rather than a
        person, no speech is coming, and waiting for it means never finishing
        the sentence that was cut off.
        """
        collected: list[np.ndarray] = []
        voiced_samples = 0
        speaking = False
        silence_started: float | None = None
        began = time.monotonic()

        while not stop_event.is_set():
            chunk = self.mini.media.get_audio_sample()
            if chunk is None or len(chunk) == 0:
                time.sleep(0.01)
                continue

            mono = self._mono(chunk)
            rms = float(np.sqrt(np.mean(np.square(mono))))
            if rms >= self.speech_rms:
                speaking = True
                silence_started = None
                voiced_samples += len(mono)
                collected.append(mono)
            elif speaking:
                # Keep trailing silence so final words aren't clipped.
                collected.append(mono)
                silence_started = silence_started or time.monotonic()
                if time.monotonic() - silence_started >= _SILENCE_TIMEOUT:
                    break

            if speaking and time.monotonic() - began > _MAX_UTTERANCE:
                break
            if (not speaking and max_wait is not None
                    and time.monotonic() - began > max_wait):
                return None

        if not collected:
            return None
        audio = np.concatenate(collected)
        if len(audio) / SAMPLE_RATE < _MIN_UTTERANCE:
            return None
        if voiced_samples / SAMPLE_RATE < _MIN_VOICED:
            # Mostly silence around a brief spike -- a door, a cough, a chair.
            return None
        return audio

    def play(self, audio: np.ndarray, stop_event: threading.Event,
             calibrate: bool = True, drain: bool = True) -> bool:
        """
        Play float32 samples, listening for the user talking over the top.

        Everything is pushed into the playback queue at once and the mic is
        watched for the expected duration; on a genuine interrupt the queue is
        dropped by restarting the output pipeline so Lancer stops mid-sentence.

        calibrate=False reuses the echo threshold measured earlier in this turn.
        Sentences are played back to back now, and re-measuring the echo floor
        on every one of them would spend a grace window per sentence -- during
        which barge-in is disabled, which is most of a short sentence.

        drain=False keeps the microphone buffer instead of discarding it. After
        an interrupt that buffer holds the first words of whatever the person
        said, and throwing it away is how the robot ends up asking them to
        repeat themselves.

        Returns True if playback was interrupted by the user.
        """
        self.speaking = True
        try:
            # The output pipeline is 2-channel; pushing mono is read as
            # interleaved stereo and plays at double speed.
            stereo = np.column_stack([audio, audio]).astype(np.float32)
            self.mini.media.push_audio_sample(stereo)
            started = time.monotonic()
            deadline = started + len(audio) / SAMPLE_RATE + 0.3
            grace_until = (started + float(self.opt("barge_grace", _BARGE_GRACE))
                           if calibrate else started)
            barge_enabled = bool(self.opt("barge_enabled", True))
            echo: list[float] = []
            threshold = self._barge_threshold or self.barge_rms
            peak = 0.0
            loud = 0
            while time.monotonic() < deadline and not stop_event.is_set():
                chunk = self.mini.media.get_audio_sample()
                if chunk is None or len(chunk) == 0:
                    time.sleep(0.01)
                    continue
                mono = self._mono(chunk)
                rms = float(np.sqrt(np.mean(np.square(mono))))

                # During the grace window Lancer is the only thing talking, so
                # whatever the mic hears is AEC residual. Calibrate against it.
                if time.monotonic() < grace_until:
                    echo.append(rms)
                    continue
                if echo:
                    measured = float(np.percentile(echo, 90))
                    threshold = max(
                        self.barge_rms,
                        measured * float(self.opt("barge_echo_headroom",
                                                  _BARGE_ECHO_HEADROOM)),
                    )
                    logger.info(
                        "Barge-in armed: echo_p90=%.4f threshold=%.4f "
                        "(speak over Lancer louder than this to interrupt)",
                        measured, threshold,
                    )
                    peak = 0.0
                    logger.debug("Barge-in threshold for this reply: %.4f", threshold)
                    echo = []
                    # Hold it for the rest of the turn's sentences.
                    self._barge_threshold = threshold

                if not barge_enabled:
                    continue
                peak = max(peak, rms)
                loud = loud + 1 if rms >= threshold else 0
                if loud >= _BARGE_CHUNKS:
                    # MediaManager has no clear_player(); dropping and restarting
                    # the output pipeline is what cuts queued audio immediately.
                    self.mini.media.stop_playing()
                    self.mini.media.start_playing()
                    logger.info("Barge-in detected -- stopping playback")
                    return True
            logger.info("Playback finished; loudest mic reading was %.4f "
                        "(threshold %.4f)", peak, threshold)
            return False
        finally:
            if drain:
                self.drain_microphone()
            self.speaking = False

    def play_reply(self, frames, stop_event: threading.Event) -> tuple:
        """
        Play a streaming reply, sentence by sentence, as it arrives.

        The server sends each sentence with its audio already synthesised, so
        Lancer starts talking while the model is still writing the rest. What
        this adds on top is the bookkeeping needed to recover from an
        interruption: the sentence that was cut off, and every sentence after
        it, is kept.

        Returns (interrupted, pending, meta):
          interrupted -- someone talked over the answer
          pending     -- audio not yet spoken, starting at the interrupted
                         sentence (from its beginning, so resuming never picks
                         up mid-word)
          meta        -- the closing frame, for logging and gestures
        """
        self._barge_threshold = 0.0
        pending: list[np.ndarray] = []
        interrupted = False
        meta: dict = {}
        first = True

        self.cues.start_thinking()
        try:
            for frame in frames:
                kind = frame.get("type")

                if kind == "status":
                    if frame.get("state") == "skill":
                        # Going out to a live source. Say so with a chirp
                        # rather than a silence people read as a hang.
                        self.cues.stop_thinking()
                        if frame.get("ok", True):
                            self.cues.skill()
                        else:
                            self.cues.error()
                        self.cues.start_thinking()
                    continue

                if kind == "sentence":
                    audio = frame.get("audio")
                    if audio is None:
                        continue
                    if interrupted:
                        # Still draining the stream so the connection closes
                        # cleanly, but nothing more gets spoken.
                        pending.append(audio)
                        continue
                    self.cues.stop_thinking()
                    was_cut = self.play(audio, stop_event,
                                        calibrate=first, drain=False)
                    first = False
                    if was_cut:
                        interrupted = True
                        # The cut sentence goes back on the queue whole.
                        pending.append(audio)
                    continue

                if kind in ("done", "error"):
                    meta = frame
        finally:
            self.cues.stop_thinking()

        if not interrupted:
            # Only now: mid-answer the buffer holds Lancer's own voice, but
            # after an interrupt it holds the first words of whoever cut in.
            self.drain_microphone()
        return interrupted, pending, meta

    def play_pending(self, pending: list, stop_event: threading.Event) -> tuple:
        """
        Finish an answer that was cut off, starting at the interrupted sentence.

        Still interruptible: someone who talks over the resumed half meant it
        the second time at least as much as the first.
        """
        for position, audio in enumerate(pending):
            if stop_event.is_set():
                break
            if self.play(audio, stop_event, calibrate=False, drain=False):
                return True, pending[position:]
        self.drain_microphone()
        return False, []

    def drain_microphone(self) -> None:
        """
        Discard buffered mic audio after playback.

        get_audio_sample() returns queued chunks, so whatever Lancer just said is
        still sitting in the buffer when listening resumes -- it would transcribe
        its own answer and reply to itself.
        """
        deadline = time.monotonic() + _POST_SPEECH_DRAIN
        dropped = 0
        while time.monotonic() < deadline:
            chunk = self.mini.media.get_audio_sample()
            if chunk is None or len(chunk) == 0:
                time.sleep(0.01)
                continue
            dropped += len(chunk)
        if dropped:
            logger.debug("Drained %.2fs of post-speech audio", dropped / SAMPLE_RATE)

    # -- direction of arrival ------------------------------------------------

    def person_recently_seen(self) -> bool:
        """True if YOLO saw a person recently -- used by the addressing gate."""
        return (time.monotonic() - self._last_person) <= _FACE_STICKY

    def vision_loop(self, stop_event: threading.Event) -> None:
        """
        Drive the head from whole-body detection, falling back to YuNet.

        While YOLO is answering we own the head via look_at_image; if the Mac
        stops responding or nobody is in frame we hand the head back to the
        daemon's own face tracking rather than freezing.
        """
        if not bool(self.opt("yolo_tracking", True)):
            logger.info("YOLO tracking disabled by config")
            return

        session = requests.Session()
        aim_u = aim_v = None          # smoothed target
        sent_u = sent_v = None        # last position actually commanded
        last_move = 0.0
        last_aim_complaint = 0.0
        while not stop_event.is_set():
            # YuNet runs on the robot with no network hop, so it is much lower
            # latency than shipping a frame to the Mac and back. Let it lead and
            # only reach for YOLO when it has actually lost the person.
            if self._face_visible():
                self._last_face = time.monotonic()
                if self.vision_mode != "yunet":
                    self.mini.start_head_tracking(
                        weight=float(self.opt("head_tracking_weight", 1.0))
                    )
                    self.vision_mode = "yunet"
                    aim_u = aim_v = sent_u = sent_v = None
                    logger.info("Vision: YuNet (face, on-device)")
                time.sleep(_TRACK_INTERVAL)
                continue

            face_gone_for = time.monotonic() - self._last_face
            if face_gone_for < _FACE_PRIMARY_GRACE:
                # Brief dropout -- YuNet holds its aim, so do not interfere.
                time.sleep(_TRACK_INTERVAL)
                continue

            time.sleep(_TRACK_INTERVAL if self.vision_mode == "yolo"
                       else _YOLO_IDLE_INTERVAL)
            if self.stopping:
                return
            if self.speaking:
                continue
            try:
                jpeg = self.mini.media.get_frame_jpeg()
                if not jpeg:
                    continue
                reply = session.post(
                    f"{SERVER_URL}/api/track",
                    files={"file": ("f.jpg", jpeg, "image/jpeg")},
                    timeout=4,
                ).json()
            except Exception:
                reply = None

            now = time.monotonic()
            if reply and reply.get("found"):
                if now - self._last_person > _AIM_RESET_AFTER or self.request_aim_reset:
                    aim_u = aim_v = sent_u = sent_v = None
                    self.request_aim_reset = False
                self._last_person = now
                if self.vision_mode != "yolo":
                    # Take the head off the daemon tracker before aiming it.
                    self.mini.stop_head_tracking()
                    self.vision_mode = "yolo"
                    logger.info("Vision: YOLO (whole body)")

                u, v = float(reply["u"]), float(reply["v"])
                if aim_u is None:
                    aim_u, aim_v = u, v
                else:
                    aim_u += _AIM_SMOOTHING * (u - aim_u)
                    aim_v += _AIM_SMOOTHING * (v - aim_v)

                deadband = _AIM_DEADBAND_FRAC * float(reply.get("width") or 640)
                moved_enough = (
                    sent_u is None
                    or math.hypot(aim_u - sent_u, aim_v - sent_v) >= deadband
                )
                if moved_enough and now - last_move >= _AIM_MIN_INTERVAL:
                    try:
                        # look_at_image would move the head itself, but it goes
                        # through look_at_world, which calls goto_target without
                        # naming a body yaw -- and that parameter defaults to
                        # 0.0, not None. Every tracking update therefore drove
                        # the body back to centre, several times a second, which
                        # is why a turn toward a voice looked like it started
                        # and then gave up. Ask it for the pose only, and issue
                        # the move with the body yaw named.
                        pose = self.mini.look_at_image(
                            int(aim_u), int(aim_v),
                            duration=_TRACK_AIM_DURATION,
                            perform_movement=False,
                        )
                        present = self._present_body_yaw()
                        self.mini.goto_target(
                            head=pose, duration=_TRACK_AIM_DURATION,
                            body_yaw=present if present is not None else None,
                        )
                        sent_u, sent_v = aim_u, aim_v
                        last_move = now
                    except Exception as e:
                        if now - last_aim_complaint > 10.0:
                            last_aim_complaint = now
                            logger.warning("Head aim failed: %s: %s",
                                           type(e).__name__, e)
            elif now - self._last_person >= _TRACK_FALLBACK_AFTER:
                if self.vision_mode != "yunet":
                    self.mini.start_head_tracking(
                        weight=float(self.opt("head_tracking_weight", 1.0))
                    )
                    self.vision_mode = "yunet"
                    logger.info("Vision: YuNet fallback (no person / server down)")

    def face_recently_seen(self) -> bool:
        """True if a face is visible now or was within _FACE_STICKY seconds."""
        if self._face_visible():
            self._last_face = time.monotonic()
            return True
        if self.person_recently_seen():
            return True
        return (time.monotonic() - self._last_face) <= _FACE_STICKY

    def _face_visible(self) -> bool:
        """True when the camera currently has a face; DoA defers to vision."""
        try:
            target = self.mini.get_tracked_face(wait=False)
            return bool(getattr(target, "detected", False))
        except Exception:
            return False

    @staticmethod
    def _normalize(angle: float) -> float:
        """Wrap a DoA bearing into [-pi, pi]; raw values overrun the yaw limit."""
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def _present_body_yaw(self) -> float | None:
        """
        Where the body is actually pointing, from the daemon.

        The SDK exposes no getter for this, but the daemon does, and it is a
        loopback request on the robot itself -- cheaper than the DoA read that
        prompts it.
        """
        try:
            response = requests.get(
                "http://127.0.0.1:8000/api/state/present_body_yaw", timeout=2
            )
            return float(response.json())
        except Exception:
            return None

    def sound_offset(self, angle: float) -> float:
        """
        Convert a DoA bearing into how far the body must turn to face it.

        The ReSpeaker reports bearings in the robot's own frame, and its
        convention is not the robot's: from the driver's own documentation,
        "0 radians is left, pi/2 radians is front/back, pi radians is right".

        The old code passed that number straight to set_target_body_yaw as an
        absolute heading. Two errors compounding: no conversion from the mic's
        frame to the robot's, and a bearing relative to where Lancer is
        currently pointing treated as an absolute world heading. Someone
        standing directly in front reads as pi/2, so Lancer would command a 90
        degree turn away from them -- which is why turning toward speech looked
        unreliable rather than simply wrong. It was wrong every time; it just
        happened to land near the speaker occasionally.

        Returns a signed offset: positive means turn one way, negative the
        other. Which way is which depends on how the array is mounted, so the
        sign is configuration rather than a constant -- flipping it is a
        dashboard change, not a redeploy.
        """
        # Not _normalize()d. That wraps into [-pi, pi], which sends pi (hard
        # right) to -pi and turns a 90 degree correction into a 270 degree one.
        # The array reports a half circle, left through front to right, so the
        # bearing is clamped to that range and used as it comes.
        bearing = min(max(float(angle), 0.0), math.pi)
        offset = (math.pi / 2) - bearing
        return float(self.opt("doa_yaw_sign", 1.0)) * offset

    def body_assist_loop(self, stop_event: threading.Event) -> None:
        """
        Let the body take over some of the head's rotation.

        Lancer looks at people almost entirely with its head. That is not a
        tuning accident: automatic body yaw is implemented as
        inverse_kinematics_safe(max_relative_yaw=65 degrees), so the body only
        contributes once the head is already 65 degrees off centre. Anyone
        standing within that cone -- which is most of a room -- gets a robot
        that swivels its head and keeps its body pointing wherever it was.

        This watches how far the head has turned and, past a much smaller
        threshold, rotates the body to absorb part of it and recentres the head
        onto the same target. The person sees the whole robot turn toward them.

        It reuses the DoA turn's pattern rather than inventing one: automatic
        yaw off, move, recentre the head, automatic yaw back on after a hold.
        That ordering matters. Automatic yaw is also what keeps the head and
        body within their mechanical limits, so it is only ever off briefly,
        and the head is recentred in the same breath -- which is precisely the
        state in which the relative angle is small and the limits are not at
        risk.
        """
        if not bool(self.opt("body_assist", False)):
            logger.info("Body assist disabled by config")
            return
        if self._auto_yaw:
            # Both enabled means both are solving for the body at once: assist
            # turns it, automatic yaw unwinds it to zero, the head swings back
            # and assist fires again. On hardware that is not a slow drift, it
            # is the robot thrashing, so refuse the combination outright rather
            # than let a config typo shake the machine apart.
            logger.warning("Body assist needs auto_body_yaw off; not starting")
            return

        complained = 0.0
        assisting = False
        reversals = 0
        last_direction = 0
        stuck = 0
        moves = 0
        settled = 0
        worst_seen = math.pi
        logger.info("Body assist on: the body follows the head past %.0f deg",
                    math.degrees(float(self.opt("body_assist_deadband",
                                                _BODY_ASSIST_DEADBAND))))

        while not stop_event.is_set():
            interval = float(self.opt("body_assist_interval", _BODY_ASSIST_INTERVAL))
            time.sleep(interval)
            try:
                enter = float(self.opt("body_assist_deadband",
                                       _BODY_ASSIST_DEADBAND))
                exit_band = enter * _BODY_ASSIST_EXIT_RATIO
                gain = float(self.opt("body_assist_gain", _BODY_ASSIST_GAIN))
                max_step = float(self.opt("body_assist_max_step",
                                          _BODY_ASSIST_MAX_STEP))

                if self.stopping:
                    return
                # Never while talking (the head is a gesture then, not a
                # tracker) and never on top of a DoA turn still settling.
                if self.speaking:
                    assisting = False
                    continue
                if time.monotonic() - self._last_turn < _DOA_MIN_INTERVAL:
                    assisting = False
                    continue

                head_yaw = self._head_yaw()
                if head_yaw is None:
                    continue
                # Once centred, forget how far off it ever was: the runaway
                # guard is about a single approach, not the whole session.
                if abs(head_yaw) < exit_band:
                    worst_seen = math.pi
                    stuck = 0
                # Hysteresis. One threshold makes the controller chatter across
                # it: nudge, drop under, stop, drift back over, nudge again.
                # Start on a clear offset, keep going until well centred.
                if not assisting and abs(head_yaw) < enter:
                    continue
                if assisting and abs(head_yaw) < exit_band:
                    assisting = False
                    settled += 1
                    continue
                assisting = True

                present = self._present_body_yaw()
                if present is None:
                    continue

                # A velocity, not a destination. The previous version jumped the
                # body most of the way in one 0.35s goto and re-centred the head
                # in the same call, which fought the tracker and looked like a
                # flinch. Small steps at a steady rate read as the robot turning
                # to face you.
                step = max(-max_step, min(max_step, head_yaw * gain))
                target = max(-_DOA_MAX_YAW, min(_DOA_MAX_YAW, present + step))
                if abs(target - present) < 0.005:
                    continue

                # Head and body move as one gesture, in a single command.
                #
                # Head pose is expressed in the body's frame, so turning the
                # body alone drags the head round with it: it swings past the
                # person by exactly the step, and the tracker then pulls it
                # back. That overshoot-and-recover on every step is what reads
                # as the head resetting while the body moves.
                #
                # Counter-rotating the head by the same angle cancels it. The
                # head holds its heading in the room while the body turns
                # underneath, which is what a person does when they turn to
                # face someone they are already looking at. The tracker still
                # owns where the head points; this only keeps that direction
                # steady through the body's move.
                pose = self._current_head_pose()
                if pose is not None:
                    self.mini.set_target(head=self._rotate_pose_z(pose, -step),
                                         body_yaw=target)
                else:
                    self.mini.set_target_body_yaw(target)
                self._last_body_move = time.monotonic()
                moves += 1
                if moves % 8 == 1:
                    logger.info("Body assist: head %+.0f deg, body %.2f -> %.2f",
                                math.degrees(head_yaw), present, target)

                # Oscillation guard. If the correction keeps changing sign the
                # controller is hunting rather than converging, and on hardware
                # that is the robot shaking. Stand down and hand the body back
                # rather than let it carry on -- a feature that turns itself off
                # is recoverable, one that shakes the machine is not.
                direction = (step > 0) - (step < 0)
                if direction and direction == -last_direction:
                    reversals += 1
                    if reversals >= _BODY_ASSIST_MAX_REVERSALS:
                        logger.error("Body assist is oscillating -- disabling it "
                                     "and returning the body to automatic yaw")
                        self.mini.set_automatic_body_yaw(True)
                        return
                elif direction:
                    reversals = max(0, reversals - 1)
                last_direction = direction or last_direction

                # Runaway guard. Rotating the body should reduce how far the
                # head is turned. If it never does, head pose is not measured
                # relative to the body and this whole control loop is chasing
                # its own tail -- stop before it winds to the limit.
                if abs(head_yaw) > worst_seen - 0.02:
                    stuck += 1
                    if stuck >= _BODY_ASSIST_MAX_STUCK:
                        logger.error("Body assist is not reducing head yaw "
                                     "(%.2f rad after %d moves) -- disabling it",
                                     head_yaw, moves)
                        self.mini.set_automatic_body_yaw(True)
                        return
                else:
                    stuck = 0
                worst_seen = min(worst_seen, abs(head_yaw))
            except Exception as e:
                if time.monotonic() - complained > 10.0:
                    complained = time.monotonic()
                    logger.warning("Body assist error: %s: %s",
                                   type(e).__name__, e)

    @staticmethod
    def _rotate_pose_z(pose, angle: float):
        """Rotate a head pose about the body's vertical axis."""
        cos, sin = math.cos(angle), math.sin(angle)
        rotation = np.array([
            [cos, -sin, 0.0, 0.0],
            [sin, cos, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        return rotation @ np.asarray(pose, dtype=float)

    def _current_head_pose(self):
        """The head's 4x4 pose in the body frame, or None if unreadable."""
        try:
            matrix = np.asarray(self.mini.get_current_head_pose(), dtype=float)
        except Exception:
            return None
        return matrix if matrix.shape == (4, 4) else None

    def _head_yaw(self) -> float | None:
        """
        How far the head is turned, in radians, relative to the body.

        Relative is the important word: this is what the body-assist loop
        drives to zero, and if it were measured in the room's frame instead
        the loop would be chasing a number its own corrections never change.
        The runaway guard exists to catch exactly that.
        """
        matrix = self._current_head_pose()
        if matrix is None:
            return None
        return float(math.atan2(matrix[1, 0], matrix[0, 0]))

    def doa_loop(self, stop_event: threading.Event) -> None:
        """
        Turn the body toward whoever is speaking.

        Head tracking handles the face once it is in frame; DoA is what gets
        Lancer facing the right way when someone speaks from off-camera.
        """
        if not bool(self.opt("doa_enabled", True)):
            return
        sustained = 0
        complained = 0.0
        while not stop_event.is_set():
            try:
                if self.stopping:
                    return
                # Give the body back to vision once the turn has had time to land.
                if (self._auto_yaw_resume_at
                        and time.monotonic() >= self._auto_yaw_resume_at):
                    self.mini.set_automatic_body_yaw(self._auto_yaw)
                    self._auto_yaw_resume_at = 0.0

                result = self.mini.media.get_DoA()
                if result is not None:
                    raw, is_speech = result
                    # How far off the sound is from where we are facing now.
                    offset = self.sound_offset(raw)
                    now = time.monotonic()
                    # Ignore DoA while speaking (AEC residual steers it), require
                    # a real change, and rate-limit so the body cannot jitter.
                    face = self._face_visible()
                    if face:
                        self._last_face = now
                    # With doa_priority the speaker wins even if a face is in
                    # frame -- Lancer should face whoever is talking to it.
                    if face and not bool(self.opt("doa_priority", True)):
                        # Sleep before looping: a bare `continue` here skipped
                        # the interval at the bottom and spun the CPU for as
                        # long as a face stayed in frame.
                        time.sleep(_DOA_INTERVAL)
                        continue

                    # A single frame of speech is a cough or a door. Require the
                    # bearing to hold before committing the body to a turn.
                    sustained = sustained + 1 if is_speech else 0

                    # These three were referenced but never assigned, so every
                    # pass through this loop raised NameError into the bare
                    # `except` below and was swallowed. The body-turn code has
                    # therefore never run once: Lancer has never turned toward a
                    # speaker, which read as a tuning problem for weeks and was
                    # not one.
                    deadband = float(self.opt("doa_deadband", _DOA_DEADBAND))
                    # A turn that lands on nobody should be retried sooner than
                    # the normal rate limit allows, or Lancer commits to a wrong
                    # bearing and sits there for a second and a half.
                    missed = (self._last_turn > 0.0
                              and now - self._last_turn >= _DOA_REACQUIRE
                              and now - self._last_face >= _DOA_REACQUIRE)
                    gap = _DOA_RETRY_INTERVAL if missed else _DOA_MIN_INTERVAL

                    if (
                        sustained >= _DOA_SUSTAIN
                        and not self.speaking
                        and (bool(self.opt("doa_priority", True))
                             or now - self._last_face >= _DOA_FACE_GRACE)
                        # Turn when the sound is off to one side of where we are
                        # pointing, rather than when a bearing differs from the
                        # last bearing we happened to act on.
                        and abs(offset) >= deadband
                        and now - self._last_turn >= gap
                    ):
                        present = self._present_body_yaw()
                        if present is None:
                            time.sleep(_DOA_INTERVAL)
                            continue
                        angle = max(-_DOA_MAX_YAW,
                                    min(_DOA_MAX_YAW, present + offset))
                        # Hand the body back to us briefly, turn, then return it
                        # to vision -- two writers at once makes the IK fail.
                        # Automatic body yaw makes the body follow the head, so
                        # switching it straight back on snapped the turn away
                        # before it finished. Hold it off while we settle.
                        self.mini.set_automatic_body_yaw(False)
                        # The head leads and the body follows, which is what a
                        # person does when a voice comes from off to one side:
                        # the head snaps round first because it is light, and
                        # the body catches up.
                        #
                        # Previously the head was centred on the body and only
                        # the body turned, so the head never appeared to react
                        # to sound at all -- it sat still until the camera
                        # happened to find a face. Now the head takes as much of
                        # the angle as it can reach and the body takes only the
                        # remainder, with body_assist bringing the rest round
                        # over the next second and unwinding the head as it goes.
                        #
                        # body_yaw is named explicitly because goto_target
                        # defaults it to 0.0, not None: omitting it drove the
                        # body to zero and undid the turn.
                        share = float(self.opt("doa_head_share", _DOA_HEAD_SHARE))
                        head_share = max(-share, min(share, offset))
                        angle = max(-_DOA_MAX_YAW,
                                    min(_DOA_MAX_YAW, present + offset - head_share))
                        self.mini.goto_target(
                            head=self._rotate_pose_z(create_head_pose(), head_share),
                            duration=0.25,
                            body_yaw=angle,
                        )
                        self.request_aim_reset = True
                        self._auto_yaw_resume_at = now + _DOA_HOLD
                        logger.info(
                            "Turned toward speech: bearing %.2f rad, %+.0f deg "
                            "from centre, head %+.0f deg, body %.2f -> %.2f%s",
                            raw, math.degrees(offset), math.degrees(head_share),
                            present, angle,
                            " (re-acquiring)" if missed else "",
                        )
                        self._last_yaw = angle
                        self._last_turn = now
                        sustained = 0
            except Exception as e:
                # Was `pass`, which is how a NameError on every single iteration
                # stayed invisible for the life of this feature. Rate-limited so
                # a genuinely noisy failure cannot flood the log, but never
                # silent again.
                if time.monotonic() - complained > 10.0:
                    complained = time.monotonic()
                    logger.warning("Sound tracking error: %s: %s",
                                   type(e).__name__, e)
            time.sleep(_DOA_INTERVAL)


def to_wav_bytes(audio: np.ndarray) -> bytes:
    """Encode float32 samples as 16-bit mono WAV for upload."""
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())
    return buf.getvalue()


def wav_bytes_to_float32(data: bytes) -> np.ndarray:
    """Decode a 16-bit mono WAV payload back to float32 samples."""
    with wave.open(io.BytesIO(data), "rb") as wav:
        frames = wav.readframes(wav.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def fetch_config() -> dict:
    """
    Pull behaviour settings from the dashboard's runtime config.

    Keeping one source of truth means the dashboard can retune the robot without
    editing code here; on any failure we fall back to the module defaults.
    """
    try:
        r = requests.get(f"{SERVER_URL}/api/config", timeout=8)
        cfg = r.json().get("config", {})
        logger.info("Loaded runtime config from server")
        return cfg
    except Exception as e:
        logger.warning("Using built-in defaults (config fetch failed: %s)", e)
        return {}


def ask_server(audio: np.ndarray, face_present: bool = False) -> dict | None:
    """
    POST one utterance to the RAG server and return its JSON reply.

    face_present tells the server whether anyone is actually looking at Lancer,
    which it uses to decide if the speech was aimed at the robot at all.
    """
    import base64

    try:
        response = requests.post(
            f"{SERVER_URL}/voice",
            params={"face": str(bool(face_present)).lower()},
            files={"file": ("utterance.wav", to_wav_bytes(audio), "audio/wav")},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as e:
        logger.error("Lancer server unreachable at %s: %s", SERVER_URL, e)
        return None

    if response.status_code in (204, 422):
        # 422 = nothing intelligible; 204 = overheard, not addressed to Lancer.
        return None
    if not response.ok:
        logger.error("Lancer server error %s: %s", response.status_code, response.text[:200])
        return None

    payload = response.json()
    payload["audio"] = wav_bytes_to_float32(base64.b64decode(payload["audio_b64"]))
    return payload


def stream_server(audio: np.ndarray, face_present: bool = False):
    """
    POST one utterance and return an iterator over the reply's frames.

    Returns None when the server declined the utterance -- 422 for nothing
    intelligible, 204 for speech that was not addressed to Lancer. Both are
    normal and both mean "say nothing", which is also exactly the test for
    whether an interruption was real: if the thing that cut Lancer off comes
    back 204 or 422, nobody was talking to it.
    """
    import base64

    try:
        response = requests.post(
            f"{SERVER_URL}/voice/stream",
            params={"face": str(bool(face_present)).lower()},
            files={"file": ("utterance.wav", to_wav_bytes(audio), "audio/wav")},
            stream=True,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as e:
        logger.error("Lancer server unreachable at %s: %s", SERVER_URL, e)
        return None

    if response.status_code in (204, 422):
        return None
    if not response.ok:
        logger.error("Lancer server error %s: %s",
                     response.status_code, response.text[:200])
        return None

    def frames():
        try:
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    frame = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Unparseable frame: %r", line[:120])
                    continue
                if frame.get("type") == "sentence" and frame.get("audio_b64"):
                    frame["audio"] = wav_bytes_to_float32(
                        base64.b64decode(frame["audio_b64"])
                    )
                yield frame
        finally:
            response.close()

    return frames()


# How long to wait, after being talked over, for the interrupter to actually say
# something. Past this it was a noise, and the answer resumes.
_RESUME_LISTEN_WAIT = 2.0


class LancerApp(ReachyMiniApp):  # type: ignore[misc]
    """Reachy Mini Apps entry point for Lancer."""

    dont_start_webserver = True

    def converse(self, lancer: "Lancer", frames, face_present: bool,
                 stop_event: threading.Event) -> None:
        """
        Speak one reply, and handle being talked over.

        Being interrupted is not the same as being asked something new. The
        board's own echo cancellation is imperfect, so a laugh, a dropped bag or
        someone greeting a friend across the room all clear the barge-in
        threshold and stop Lancer mid-sentence. Previously that was the end of
        the answer: the rest was discarded and the robot went back to idle
        having said half a thing.

        So after an interrupt, find out whether anyone was actually talking to
        it. Two tests, both of which already existed:

          - nothing intelligible follows within a couple of seconds, or the
            utterance fails the capture gates (too short, mostly silence)
          - the server's addressing gate returns 204 (not aimed at Lancer) or
            422 (not speech)

        Either means the interruption was not real, and the answer picks up from
        the beginning of the sentence it was cut off in. Anything else is a real
        question and gets answered instead.
        """
        resume_enabled = bool(lancer.opt("resume_after_false_barge", True))

        while frames is not None:
            interrupted, pending, meta = lancer.play_reply(frames, stop_event)
            frames = None

            if meta.get("type") == "error":
                logger.error("Server reported: %s", meta.get("detail"))
                return
            if meta:
                logger.info("heard=%r answer=%r skills=%s",
                            meta.get("heard", ""), meta.get("answer", "")[:120],
                            meta.get("skills") or [])
            lancer.confused() if meta.get("unknown") else lancer.answering()

            # Each pass here is one interruption: work out whether it was real,
            # and either resume what was cut off or go answer the new question.
            while (interrupted and pending and resume_enabled
                   and not stop_event.is_set()):
                follow_up = lancer.listen(stop_event, max_wait=_RESUME_LISTEN_WAIT)

                if follow_up is not None:
                    reply = stream_server(follow_up, lancer.face_recently_seen())
                    if reply is not None:
                        # A real question. The unfinished answer is dropped on
                        # purpose -- nobody wants the end of the previous
                        # answer after asking something else.
                        logger.info("Interruption was a real question")
                        lancer.thinking()
                        frames = reply
                        break
                    reason = "not addressed to Lancer"
                else:
                    reason = "a noise, not a person"

                logger.info("Interruption was %s -- resuming %d sentence(s)",
                            reason, len(pending))
                interrupted, pending = lancer.play_pending(pending, stop_event)

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        pick_server()
        cfg = fetch_config()
        lancer = Lancer(reachy_mini, cfg)

        # wake_up() plays the pose and sound but does NOT enable torque -- without
        # this the head flops back into the shell and every command still reports
        # success. Motors first, then wake.
        reachy_mini.enable_motors()
        reachy_mini.wake_up()
        # wake_up() leaves the antennas vertical, which is the unstable point
        # where they visibly shake. Bias them off vertical straight away.
        lancer.answering()
        # get_audio_sample() returns None until the audio device is recording.
        reachy_mini.acquire_media()
        reachy_mini.media.start_recording()
        # push_audio_sample() needs the output pipeline running first.
        try:
            reachy_mini.media.start_playing()
        except Exception as e:
            logger.warning("start_playing() failed (may auto-start): %s", e)
        lancer.calibrate()
        # Head follows the face; the body follows the head, so Lancer turns to
        # face people rather than craning at them. Tracking stays on throughout.
        # start_head_tracking() only enables the aiming loop -- the face detector
        # itself is separate, and without it the tracker never sees anyone and the
        # head just recenters on the lost-target timeout.
        try:
            requests.post(f"http://127.0.0.1:8000/api/media/tracking/enable",
                          json={}, timeout=8)
            logger.info("Face detector enabled")
        except Exception as e:
            logger.warning("Could not enable face detector: %s", e)
        reachy_mini.start_head_tracking(
            weight=float(cfg.get("head_tracking_weight") or 1.0)
        )
        reachy_mini.set_automatic_body_yaw(bool(cfg.get("auto_body_yaw", True)))

        if not bool(cfg.get("doa_enabled", True)):
            logger.info("Audio direction tracking disabled by config")
        doa_thread = threading.Thread(
            target=lancer.doa_loop, args=(stop_event,), daemon=True
        )
        doa_thread.start()
        vision_thread = threading.Thread(
            target=lancer.vision_loop, args=(stop_event,), daemon=True
        )
        vision_thread.start()
        body_thread = threading.Thread(
            target=lancer.body_assist_loop, args=(stop_event,), daemon=True
        )
        body_thread.start()
        logger.info("Audio direction tracking: on (deadband %.2f rad)", _DOA_DEADBAND)
        logger.info("Lancer ready -- server at %s", SERVER_URL)

        try:
            while not stop_event.is_set():
                audio = lancer.listen(stop_event)
                if audio is None:
                    continue

                logger.info("Captured %.1fs of speech", len(audio) / SAMPLE_RATE)

                # Ask first, and only disturb the pose once there is something to
                # say. Gesturing on every noise trigger is what made Lancer twitch.
                # A face that was visible moments ago still counts: detection
                # drops out constantly when someone turns their head to speak.
                face_present = lancer.face_recently_seen()
                lancer.cues.heard()
                frames = stream_server(audio, face_present)
                if frames is None:
                    # Overheard chatter: say nothing, move nothing.
                    continue
                lancer.thinking()

                try:
                    self.converse(lancer, frames, face_present, stop_event)
                finally:
                    lancer.answering()
        finally:
            # Take the head away from everything that steers it before folding
            # the robot up. Each tracking loop sleeps a fraction of a second
            # between checks, so simply setting stop_event leaves them able to
            # issue one more move -- landing in the middle of goto_sleep() and
            # fighting it, which is the head twitching on the way down.
            lancer.stopping = True
            for thread in (doa_thread, vision_thread, body_thread):
                try:
                    thread.join(timeout=1.5)
                except Exception:
                    pass
            try:
                reachy_mini.set_automatic_body_yaw(False)
            except Exception:
                pass
            reachy_mini.stop_head_tracking()
            # The daemon's own face tracker is a separate thing from the app's
            # loops and keeps driving the head after they are gone. The app
            # turns it on at startup, so it turns it off here.
            try:
                requests.post("http://127.0.0.1:8000/api/media/tracking/disable",
                              json={}, timeout=4)
            except Exception as e:
                logger.debug("Could not disable face detector: %s", e)
            for cleanup in (
                reachy_mini.media.stop_recording,
                reachy_mini.goto_sleep,
                reachy_mini.disable_motors,
            ):
                try:
                    cleanup()
                except Exception:
                    pass


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    app = LancerApp()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
