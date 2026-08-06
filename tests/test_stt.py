"""The speech-to-text layer, without a model and without a microphone.

Everything here runs on a developer machine with no sherpa-onnx wheel, no ONNX
graph and no whisper-server. That is the point: accuracy is a question about
real speech on the Pi and `scripts/stt_test.py` is what answers it, while the
things tested here — language folding, guards against bad audio, whether a
failed transcription can take the assistant down — are decidable at a desk and
are exactly what nobody wants to discover by talking to the device.

The SenseVoice recogniser is faked. Faking sherpa-onnx rather than skipping
without it means the *interesting* half is covered on every run: the fake
returns what the real one returns — a text field and a `<|yue|>`-style language
tag — and the code under test is the code that turns those into a Transcript.
"""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from aia.core.config import CONFIG, SttConfig
from aia.stt import build, detect_script, parse_lang_tag, strip_meta
from aia.stt.base import SttUnavailable, Transcript, as_float32
from aia.stt.sensevoice import SenseVoiceSTT

RATE = CONFIG.audio.target_rate


class FakeResult:
    def __init__(self, text: str, lang: str = ""):
        self.text = text
        self.lang = lang


class FakeStream:
    def __init__(self, result):
        self.result = result
        self.accepted: list[tuple[int, np.ndarray]] = []

    def accept_waveform(self, rate, samples):
        self.accepted.append((rate, samples))


class FakeRecognizer:
    """Stands in for sherpa_onnx.OfflineRecognizer.

    `raises` makes the recogniser throw the way a native library does — which
    is the case the assistant must survive, and the one that cannot be
    provoked from a real model on demand.
    """

    def __init__(self, text="", lang="", raises=None):
        self.result = FakeResult(text, lang)
        self.raises = raises
        self.streams: list[FakeStream] = []

    def create_stream(self):
        stream = FakeStream(self.result)
        self.streams.append(stream)
        return stream

    def decode_stream(self, stream):
        if self.raises is not None:
            raise self.raises


def sensevoice(text="", lang="", raises=None, cfg=None) -> SenseVoiceSTT:
    """A backend with its model already 'loaded', so nothing touches disk."""
    stt = SenseVoiceSTT(cfg or replace(CONFIG.stt, backend="sensevoice"), RATE)
    stt._recognizer = FakeRecognizer(text, lang, raises)
    return stt


def speech(ms: int = 1000) -> np.ndarray:
    """Audio of a given length. Content is irrelevant — the model is fake."""
    return np.zeros(RATE * ms // 1000, dtype=np.int16)


class Script(unittest.TestCase):
    """`detect_script` reads script, and must not be asked to read language."""

    def test_english_and_mandarin(self):
        self.assertEqual(detect_script("search lyrics"), "en")
        self.assertEqual(detect_script("搜索歌词"), "zh")

    def test_code_switched_resolves_to_the_carrier_language(self):
        # The sentence is English and only the proper noun is not, so an
        # English reply is the right one.
        self.assertEqual(detect_script("Play 周杰伦"), "en")

    def test_script_counting_misreads_a_long_english_proper_noun(self):
        # Pinned as the known limit of counting characters, not endorsed. The
        # carrier language here is Mandarin — 搜索…的歌词 is the command — but
        # "TaylorSwift" is eleven Latin characters against five Han, so this
        # says English.
        #
        # It is unchanged behaviour and it is no longer load-bearing: under
        # SenseVoice this function is consulted third, after the model's own
        # verdict, and the model hears Mandarin. See the Language tests.
        self.assertEqual(detect_script("搜索 Taylor Swift 的歌词"), "en")

    def test_cantonese_is_han_and_reads_as_zh(self):
        # This is correct and is the whole reason Cantonese cannot be detected
        # from text. 搵 and 嘅 are Cantonese-specific characters and they are
        # still Han; only the recogniser, listening to the audio, can tell.
        self.assertEqual(detect_script("帮我搵下歌词"), "zh")

    def test_unsupported_scripts(self):
        for text in ("총치", "よいしょ", "привет"):
            self.assertEqual(detect_script(text), "other", text)

    def test_empty(self):
        self.assertIsNone(detect_script(""))


class Meta(unittest.TestCase):
    def test_sensevoice_tags_are_removed_from_a_transcript(self):
        self.assertEqual(
            strip_meta("<|zh|><|NEUTRAL|><|Speech|><|withitn|>搜索歌词"),
            "搜索歌词")

    def test_text_without_tags_is_untouched(self):
        self.assertEqual(strip_meta("search lyrics"), "search lyrics")

    def test_the_language_field_is_parsed_not_stripped(self):
        # These two functions look interchangeable and are opposites, and
        # using the wrong one here cost the backend its entire reason for
        # existing: strip_meta("<|yue|>") is the empty string, so Cantonese
        # detection silently degraded to guessing from Han characters, which
        # cannot tell Cantonese from Mandarin at all.
        self.assertEqual(strip_meta("<|yue|>"), "")
        self.assertEqual(parse_lang_tag("<|yue|>"), "yue")

    def test_a_bare_code_and_an_absent_one(self):
        self.assertEqual(parse_lang_tag("zh"), "zh")
        self.assertIsNone(parse_lang_tag(""))
        self.assertIsNone(parse_lang_tag(None))


class Float32(unittest.TestCase):
    def test_most_negative_sample_cannot_exceed_minus_one(self):
        # Dividing by 32767 would put this at -1.00003, and a downstream clamp
        # would flatten the loudest part of the loudest utterance.
        self.assertGreaterEqual(as_float32(np.array([-32768], dtype=np.int16))[0], -1.0)

    def test_scale(self):
        out = as_float32(np.array([0, 16384], dtype=np.int16))
        self.assertAlmostEqual(out[0], 0.0)
        self.assertAlmostEqual(out[1], 0.5)


class Build(unittest.TestCase):
    def test_selects_the_configured_backend(self):
        stt = build(replace(CONFIG.stt, backend="sensevoice"), RATE)
        self.assertIsInstance(stt, SenseVoiceSTT)

    def test_case_and_whitespace_are_tolerated(self):
        self.assertIsInstance(build(replace(CONFIG.stt, backend=" SenseVoice "), RATE),
                              SenseVoiceSTT)

    def test_an_unknown_backend_raises_rather_than_defaulting(self):
        # A typo that silently ran the old engine would present as "the new
        # recogniser is no better", which is the most expensive possible way
        # to find out.
        with self.assertRaises(SttUnavailable):
            build(replace(CONFIG.stt, backend="sensevoise"), RATE)


class Language(unittest.TestCase):
    """What was heard, versus what AIA can answer in."""

    def test_mandarin(self):
        result = sensevoice("搜索歌词", "<|zh|>").listen(speech())
        self.assertEqual(result.text, "搜索歌词")
        self.assertEqual(result.detected, "zh")
        self.assertEqual(result.language, "zh")

    def test_cantonese_is_recognised_as_yue_and_answered_in_mandarin(self):
        # The two halves of the Cantonese decision, in one assertion each.
        # Piper has no yue voice, so the reply is Mandarin — but the fact that
        # Cantonese was understood *as Cantonese* has to survive into the
        # journal, or the only visible evidence is a Mandarin reply.
        result = sensevoice("帮我搵下歌词", "<|yue|>").listen(speech())
        self.assertEqual(result.detected, "yue")
        self.assertEqual(result.language, "zh")

    def test_english(self):
        result = sensevoice("search lyrics", "<|en|>").listen(speech())
        self.assertEqual(result.language, "en")

    def test_japanese_and_korean_fall_back_to_mandarin(self):
        # SenseVoice knows five languages and AIA supports two of them. On
        # this device a ja or ko tag on a real utterance means Mandarin was
        # misheard, not that anyone switched to Japanese.
        for tag in ("<|ja|>", "<|ko|>"):
            self.assertEqual(sensevoice("なに", tag).listen(speech()).language, "zh")

    def test_a_bare_tag_without_pipes_is_understood(self):
        # Different sherpa-onnx builds report "<|zh|>" or "zh".
        self.assertEqual(sensevoice("搜索歌词", "zh").listen(speech()).detected, "zh")

    def test_falls_back_to_script_when_nothing_is_reported(self):
        self.assertEqual(sensevoice("search lyrics", "").listen(speech()).language, "en")
        self.assertEqual(sensevoice("搜索歌词", "").listen(speech()).language, "zh")

    def test_the_transcript_is_not_translated(self):
        # The spec is explicit: what the router receives is what was said.
        mixed = "搜索 Taylor Swift 的歌词"
        self.assertEqual(sensevoice(mixed, "<|zh|>").listen(speech()).text, mixed)

    def test_the_model_verdict_beats_character_counting_on_mixed_speech(self):
        # The case `Script.test_script_counting_misreads_a_long_english_proper_noun`
        # pins: text whose Latin characters outnumber its Han, spoken in
        # Mandarin. `detect_script` says "en" and the model says zh, and the
        # model is right — it heard the sentence, and asking it is the reason
        # the reply comes back in the language the person was speaking.
        result = sensevoice("搜索 Taylor Swift 的歌词", "<|zh|>").listen(speech())
        self.assertEqual(detect_script(result.text), "en")
        self.assertEqual(result.language, "zh")

    def test_a_named_language_is_honoured_when_the_model_reports_nothing(self):
        # The confirmation path: the assistant asked a question in Mandarin and
        # is holding the floor for the answer.
        result = sensevoice("确定", "").listen(speech(), language="zh")
        self.assertEqual(result.language, "zh")


class BadAudio(unittest.TestCase):
    """None of these may raise. All of them mean 'apologise and keep listening'."""

    def test_empty_audio(self):
        result = sensevoice("should not be reached").listen(np.zeros(0, dtype=np.int16))
        self.assertEqual(result.text, "")
        self.assertFalse(result)

    def test_audio_too_short_to_hold_a_phoneme(self):
        result = sensevoice("should not be reached").listen(speech(10))
        self.assertEqual(result.text, "")

    def test_none(self):
        self.assertEqual(sensevoice("x").listen(None).text, "")

    def test_a_failure_inside_the_recogniser_is_a_dead_turn_not_a_crash(self):
        stt = sensevoice(raises=RuntimeError("onnxruntime exploded"))
        with self.assertLogs("aia.stt.sensevoice", level="ERROR"):
            result = stt.listen(speech())
        self.assertEqual(result.text, "")
        self.assertFalse(result)

    def test_a_missing_model_is_reported_by_wait_ready_not_raised(self):
        # main() turns False into one clear line and a non-zero exit, which is
        # what a person needs at startup. A traceback out of the constructor
        # would be a restart loop instead.
        cfg = replace(CONFIG.stt, backend="sensevoice",
                      sensevoice=replace(CONFIG.stt.sensevoice,
                                         directory=CONFIG.stt.sensevoice.directory / "nope"))
        stt = SenseVoiceSTT(cfg, RATE)
        with self.assertLogs("aia.stt.sensevoice", level="ERROR"):
            self.assertFalse(stt.wait_ready(timeout=1))


class Rate(unittest.TestCase):
    def test_the_true_sample_rate_reaches_the_recogniser(self):
        # Not the model's 16 kHz — the audio's. sherpa-onnx resamples on this
        # number, so a lie here produces the same class of confident nonsense
        # that a wrong WAV header produced under whisper.
        stt = sensevoice("x", "<|en|>")
        stt.listen(speech())
        rate, _ = stt._recognizer.streams[0].accepted[0]
        self.assertEqual(rate, RATE)


class Reporting(unittest.TestCase):
    def test_repr_shows_the_spoken_language_when_it_differs(self):
        self.assertIn("zh/yue", repr(Transcript("歌词", "zh", 120.0, detected="yue")))

    def test_repr_does_not_repeat_the_language_when_they_agree(self):
        self.assertNotIn("zh/zh", repr(Transcript("歌词", "zh", 120.0, detected="zh")))

    def test_confidence_is_none_rather_than_zero_on_this_backend(self):
        # SenseVoice exposes no per-word probabilities. None means "unknown";
        # 0.0 would mean "certainly wrong" to anything that reads it.
        self.assertIsNone(sensevoice("搜索歌词", "<|zh|>").listen(speech()).confidence)

    def test_describe_names_the_engine_and_thread_count(self):
        described = sensevoice("x").describe()
        self.assertIn("SenseVoice", described["engine"])
        self.assertEqual(described["num_threads"], CONFIG.stt.sensevoice.num_threads)


class ConfigInvariants(unittest.TestCase):
    def test_whisper_capture_cap_is_not_applied_to_sensevoice(self):
        # SenseVoice has no fixed encoder window, so a capture longer than
        # whisper's readable audio is not an error for it.
        from aia.core.config import Config, VadConfig
        cfg = Config(stt=SttConfig(backend="sensevoice"),
                     vad=VadConfig(max_utterance_ms=20000))
        self.assertEqual(cfg.vad.max_utterance_ms, 20000)

    def test_whisper_still_enforces_it(self):
        from aia.core.config import Config, VadConfig
        with self.assertRaises(ValueError):
            Config(stt=SttConfig(backend="whisper"),
                   vad=VadConfig(max_utterance_ms=20000))

    def test_the_default_backend_is_sensevoice(self):
        # Guards against the environment override leaking into a test run and
        # against the default quietly reverting.
        self.assertEqual(SttConfig().backend, "sensevoice")

    def test_reply_languages_exclude_cantonese(self):
        # Adding yue here without adding a Piper voice would route a Cantonese
        # turn to whichever voice happens to be first in the dict.
        self.assertNotIn("yue", CONFIG.stt.supported_languages)
        self.assertIn("yue", CONFIG.stt.sensevoice.recognised_languages)


if __name__ == "__main__":
    unittest.main()
