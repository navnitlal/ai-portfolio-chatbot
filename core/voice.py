"""Voice helpers: recording (sounddevice), speech-to-text (Whisper), and text-to-speech (TTS-1).

Uses the OpenAI Python SDK for STT/TTS and sounddevice for mic capture.
No Streamlit or LangGraph dependencies.
"""
from __future__ import annotations

import io
import tempfile
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
from openai import OpenAI

from observability import logger

# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

def record_audio(
    max_duration: float = 10.0,
    sample_rate: int = 16000,
    silence_threshold: float = 0.01,
    silence_duration: float = 1.5,
) -> bytes | None:
    """Record from the default mic, auto-stopping on silence.

    Args:
        max_duration: Hard cap in seconds.
        sample_rate: Sample rate in Hz (16 kHz is fine for Whisper).
        silence_threshold: RMS energy below which audio counts as silence.
        silence_duration: Seconds of continuous silence before auto-stop.

    Returns:
        WAV bytes, or *None* if nothing was captured.
    """
    blocksize = 1024
    frames: list[np.ndarray] = []
    max_silence_blocks = int(silence_threshold and silence_duration * sample_rate / blocksize)
    silence_count = 0
    speaking_started = False
    stop_flag = False

    def _callback(indata, frame_count, time_info, status):
        nonlocal silence_count, speaking_started, stop_flag
        frames.append(indata.copy())
        energy = float(np.sqrt(np.mean(indata ** 2)))
        if energy >= silence_threshold:
            speaking_started = True
            silence_count = 0
        else:
            silence_count += 1
        if speaking_started and silence_count > max_silence_blocks:
            stop_flag = True

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocksize=blocksize,
            callback=_callback,
        ):
            elapsed = 0.0
            step = 0.1  # check every 100 ms
            while elapsed < max_duration and not stop_flag:
                sd.sleep(int(step * 1000))
                elapsed += step
    except Exception as exc:
        logger.error("sounddevice recording failed: %s", exc)
        return None

    if not frames:
        return None

    audio_data = np.concatenate(frames)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    buf.seek(0)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Transcription (Whisper)
# ---------------------------------------------------------------------------

def transcribe(audio_bytes: bytes, api_key: str) -> str:
    """Write *audio_bytes* to a temp .wav file and transcribe via Whisper.

    Returns the transcribed text, or an empty string on failure.
    """
    try:
        tmp = Path(tempfile.mktemp(suffix=".wav"))
        tmp.write_bytes(audio_bytes)
        client = OpenAI(api_key=api_key)
        with open(tmp, "rb") as f:
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
            )
        return result.text.strip()
    except Exception as exc:
        logger.error("Whisper transcription failed: %s", exc)
        return ""
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Text-to-Speech (TTS-1)
# ---------------------------------------------------------------------------

def synthesize(text: str, api_key: str, voice: str = "alloy") -> bytes | None:
    """Convert *text* to speech via TTS-1, returning MP3 bytes (or None on failure).

    Text is truncated to 4 096 characters (the TTS-1 limit).
    """
    if not text:
        return None
    text = text[:4096]
    try:
        client = OpenAI(api_key=api_key)
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
        )
        return response.content
    except Exception as exc:
        logger.error("TTS synthesis failed: %s", exc)
        return None
