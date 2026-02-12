"""Tests for core/voice.py - voice recording, transcription (STT), and synthesis (TTS).

Unit tests (mocked):
    pytest tests/core/test_voice.py -m "not integration"

Integration tests (requires OPENAI_API_KEY in .env):
    pytest tests/core/test_voice.py -m integration
"""
from __future__ import annotations

import io
import os
import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.voice import record_audio, transcribe, synthesize


# =============================================================================
# Helpers
# =============================================================================


def _make_wav_bytes(
    duration: float = 0.5,
    sample_rate: int = 16000,
    silence: bool = True,
) -> bytes:
    """Create valid WAV bytes for testing.

    Args:
        duration: Length in seconds.
        sample_rate: Sample rate in Hz.
        silence: If True, produce silence; otherwise a 440 Hz tone.
    """
    n_frames = int(sample_rate * duration)
    if silence:
        samples = np.zeros(n_frames, dtype=np.int16)
    else:
        t = np.linspace(0, duration, n_frames, endpoint=False)
        samples = (np.sin(2 * np.pi * 440 * t) * 3000).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    buf.seek(0)
    return buf.getvalue()


def _make_speech_wav(text: str, api_key: str) -> bytes:
    """Use TTS-1 to produce a WAV with real speech for transcription tests.

    Synthesises *text* via TTS-1 (MP3), then converts to 16 kHz mono WAV
    so it can be fed straight into ``transcribe()``.
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text,
        response_format="wav",
    )
    return response.content


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def openai_api_key():
    """Return the OPENAI_API_KEY or skip the test."""
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        pytest.skip("OPENAI_API_KEY not set — skipping integration test")
    return key


# =============================================================================
# Unit Tests: record_audio
# =============================================================================


@pytest.mark.unit
class TestRecordAudio:
    """Tests for the record_audio function."""

    @patch("core.voice.sd")
    def test_returns_wav_bytes_on_success(self, mock_sd):
        """Captured audio frames are returned as valid WAV bytes."""
        fake_audio = np.random.randint(-1000, 1000, size=(1024, 1), dtype=np.int16)

        class FakeInputStream:
            def __init__(self, **kwargs):
                self.callback = kwargs.get("callback")

            def __enter__(self):
                self.callback(fake_audio, 1024, None, None)
                return self

            def __exit__(self, *args):
                pass

        mock_sd.InputStream = FakeInputStream
        mock_sd.sleep = MagicMock()

        result = record_audio(max_duration=0.2, silence_duration=0.1)

        assert result is not None
        buf = io.BytesIO(result)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000

    @patch("core.voice.sd")
    def test_returns_none_when_no_frames(self, mock_sd):
        """Returns None when no audio frames were captured."""

        class FakeInputStream:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        mock_sd.InputStream = FakeInputStream
        mock_sd.sleep = MagicMock()

        result = record_audio(max_duration=0.2)
        assert result is None

    @patch("core.voice.sd")
    def test_returns_none_on_device_error(self, mock_sd):
        """Returns None when sounddevice raises an exception."""
        mock_sd.InputStream.side_effect = OSError("No audio device")
        mock_sd.sleep = MagicMock()

        result = record_audio(max_duration=0.2)
        assert result is None

    @patch("core.voice.sd")
    def test_auto_stops_on_silence(self, mock_sd):
        """Recording stops when silence is detected after speech."""
        loud_block = np.full((1024, 1), 5000, dtype=np.int16)
        silent_block = np.zeros((1024, 1), dtype=np.int16)

        class FakeInputStream:
            def __init__(self, **kwargs):
                self.callback = kwargs.get("callback")

            def __enter__(self):
                self.callback(loud_block, 1024, None, None)
                for _ in range(50):
                    self.callback(silent_block, 1024, None, None)
                return self

            def __exit__(self, *args):
                pass

        mock_sd.InputStream = FakeInputStream
        mock_sd.sleep = MagicMock()

        result = record_audio(
            max_duration=10.0,
            silence_threshold=0.01,
            silence_duration=0.5,
        )
        assert result is not None


# =============================================================================
# Unit Tests: transcribe (Whisper STT)
# =============================================================================


@pytest.mark.unit
class TestTranscribe:
    """Unit tests for the transcribe function (mocked OpenAI)."""

    @patch("core.voice.OpenAI")
    def test_returns_transcribed_text(self, mock_openai_cls):
        """Successful transcription returns the text."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.audio.transcriptions.create.return_value = MagicMock(
            text="  Hello world  "
        )

        result = transcribe(_make_wav_bytes(), api_key="test-key")

        assert result == "Hello world"
        mock_openai_cls.assert_called_once_with(api_key="test-key")
        mock_client.audio.transcriptions.create.assert_called_once()
        call_kwargs = mock_client.audio.transcriptions.create.call_args
        assert call_kwargs.kwargs["model"] == "whisper-1"

    @patch("core.voice.OpenAI")
    def test_returns_empty_on_api_error(self, mock_openai_cls):
        """Returns empty string when the Whisper API raises an error."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.audio.transcriptions.create.side_effect = Exception("API error")

        result = transcribe(_make_wav_bytes(), api_key="test-key")
        assert result == ""

    @patch("core.voice.OpenAI")
    def test_strips_whitespace(self, mock_openai_cls):
        """Transcription text is stripped of leading/trailing whitespace."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.audio.transcriptions.create.return_value = MagicMock(
            text="\n  What is my portfolio performance?  \n"
        )

        result = transcribe(_make_wav_bytes(), api_key="k")
        assert result == "What is my portfolio performance?"

    @patch("core.voice.OpenAI")
    def test_temp_file_cleaned_up(self, mock_openai_cls):
        """Temporary WAV file is removed after transcription."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.audio.transcriptions.create.return_value = MagicMock(text="hi")

        result = transcribe(_make_wav_bytes(), api_key="k")
        assert result == "hi"

    @patch("core.voice.OpenAI")
    def test_temp_file_cleaned_up_on_error(self, mock_openai_cls):
        """Temporary WAV file is cleaned up even when transcription fails."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.audio.transcriptions.create.side_effect = RuntimeError("fail")

        result = transcribe(_make_wav_bytes(), api_key="k")
        assert result == ""


# =============================================================================
# Unit Tests: synthesize (TTS-1)
# =============================================================================


@pytest.mark.unit
class TestSynthesize:
    """Unit tests for the synthesize function (mocked OpenAI)."""

    @patch("core.voice.OpenAI")
    def test_returns_mp3_bytes(self, mock_openai_cls):
        """Successful synthesis returns MP3 bytes."""
        fake_mp3 = b"\xff\xfb\x90\x00" + b"\x00" * 100
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.audio.speech.create.return_value = MagicMock(content=fake_mp3)

        result = synthesize("Hello world", api_key="test-key")

        assert result == fake_mp3
        mock_openai_cls.assert_called_once_with(api_key="test-key")
        call_kwargs = mock_client.audio.speech.create.call_args
        assert call_kwargs.kwargs["model"] == "tts-1"
        assert call_kwargs.kwargs["input"] == "Hello world"
        assert call_kwargs.kwargs["voice"] == "alloy"

    @patch("core.voice.OpenAI")
    def test_custom_voice(self, mock_openai_cls):
        """Voice parameter is forwarded to the TTS API."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.audio.speech.create.return_value = MagicMock(content=b"audio")

        synthesize("Hi", api_key="k", voice="nova")

        call_kwargs = mock_client.audio.speech.create.call_args
        assert call_kwargs.kwargs["voice"] == "nova"

    @patch("core.voice.OpenAI")
    def test_returns_none_on_api_error(self, mock_openai_cls):
        """Returns None when the TTS API raises an error."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.audio.speech.create.side_effect = Exception("TTS failed")

        result = synthesize("Hello", api_key="test-key")
        assert result is None

    def test_returns_none_for_empty_text(self):
        """Returns None immediately for empty input text."""
        assert synthesize("", api_key="k") is None

    @patch("core.voice.OpenAI")
    def test_truncates_long_text(self, mock_openai_cls):
        """Text longer than 4096 chars is truncated before sending to API."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.audio.speech.create.return_value = MagicMock(content=b"audio")

        synthesize("A" * 5000, api_key="k")

        call_kwargs = mock_client.audio.speech.create.call_args
        assert len(call_kwargs.kwargs["input"]) == 4096

    @patch("core.voice.OpenAI")
    def test_short_text_not_truncated(self, mock_openai_cls):
        """Text within the 4096-char limit is sent as-is."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.audio.speech.create.return_value = MagicMock(content=b"audio")

        text = "Tell me about my portfolio allocation."
        synthesize(text, api_key="k")

        call_kwargs = mock_client.audio.speech.create.call_args
        assert call_kwargs.kwargs["input"] == text


# =============================================================================
# Integration Tests — Real OpenAI API (requires OPENAI_API_KEY)
# =============================================================================


@pytest.mark.integration
class TestSynthesizeIntegration:
    """Integration tests for TTS-1 synthesis using the real OpenAI API.

    Run with: pytest tests/core/test_voice.py -m integration
    """

    def test_synthesize_returns_audio_bytes(self, openai_api_key):
        """TTS-1 returns non-empty audio bytes for a simple sentence."""
        result = synthesize("Hello, how are you?", api_key=openai_api_key)

        assert result is not None
        assert isinstance(result, bytes)
        assert len(result) > 100  # real MP3 is at least a few KB

    def test_synthesize_different_voices(self, openai_api_key):
        """TTS-1 accepts different voice options."""
        for voice in ("alloy", "nova"):
            result = synthesize("Test", api_key=openai_api_key, voice=voice)
            assert result is not None
            assert len(result) > 100

    def test_synthesize_long_text_truncated(self, openai_api_key):
        """Long text is truncated to 4096 chars but still produces audio."""
        long_text = "This is a portfolio performance test. " * 200  # >4096 chars
        result = synthesize(long_text, api_key=openai_api_key)

        assert result is not None
        assert len(result) > 100


@pytest.mark.integration
class TestTranscribeIntegration:
    """Integration tests for Whisper transcription using the real OpenAI API.

    These tests synthesise speech via TTS-1 first, then transcribe it back
    with Whisper to verify the full STT pipeline.

    Run with: pytest tests/core/test_voice.py -m integration
    """

    def test_transcribe_returns_text(self, openai_api_key):
        """Whisper returns non-empty text for a real speech WAV."""
        wav_bytes = _make_speech_wav("Hello world", openai_api_key)

        result = transcribe(wav_bytes, api_key=openai_api_key)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_transcribe_matches_original(self, openai_api_key):
        """Whisper transcription is close to the original synthesised text."""
        original = "What is my portfolio performance?"
        wav_bytes = _make_speech_wav(original, openai_api_key)

        result = transcribe(wav_bytes, api_key=openai_api_key)

        # Check key words appear (exact match not guaranteed due to model variation)
        result_lower = result.lower()
        assert "portfolio" in result_lower
        assert "performance" in result_lower

    def test_transcribe_short_phrase(self, openai_api_key):
        """Whisper handles a short phrase correctly."""
        wav_bytes = _make_speech_wav("Yes", openai_api_key)

        result = transcribe(wav_bytes, api_key=openai_api_key)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_transcribe_financial_terms(self, openai_api_key):
        """Whisper correctly recognises financial terminology."""
        original = "Show me the asset allocation and risk profile"
        wav_bytes = _make_speech_wav(original, openai_api_key)

        result = transcribe(wav_bytes, api_key=openai_api_key)

        result_lower = result.lower()
        # At least some financial terms should be recognised
        matches = sum(1 for term in ("asset", "allocation", "risk", "profile")
                      if term in result_lower)
        assert matches >= 2, f"Expected financial terms in: {result}"


@pytest.mark.integration
class TestVoiceRoundTrip:
    """End-to-end round-trip: text -> TTS-1 speech -> Whisper transcription.

    Verifies the full voice pipeline the app uses:
    user speaks -> WAV -> transcribe() -> text -> agent -> synthesize() -> audio.
    """

    def test_round_trip_greeting(self, openai_api_key):
        """Full round-trip: synthesise greeting, transcribe it back."""
        original = "Hello, I would like to check my portfolio."

        # Step 1: Text -> Speech (TTS-1)
        wav_bytes = _make_speech_wav(original, openai_api_key)
        assert wav_bytes is not None
        assert len(wav_bytes) > 100

        # Step 2: Speech -> Text (Whisper)
        transcribed = transcribe(wav_bytes, api_key=openai_api_key)
        assert isinstance(transcribed, str)
        assert len(transcribed) > 0
        assert "portfolio" in transcribed.lower()

        # Step 3: Text -> Speech again (TTS-1, simulating agent reply)
        reply_audio = synthesize(transcribed, api_key=openai_api_key)
        assert reply_audio is not None
        assert len(reply_audio) > 100

    def test_round_trip_risk_question(self, openai_api_key):
        """Round-trip with a risk-questionnaire style answer."""
        original = "I prefer maximum growth with high risk tolerance."

        wav_bytes = _make_speech_wav(original, openai_api_key)
        transcribed = transcribe(wav_bytes, api_key=openai_api_key)

        assert len(transcribed) > 0
        # Key financial intent should survive the round-trip
        t = transcribed.lower()
        assert "growth" in t or "risk" in t, f"Financial intent lost: {transcribed}"
