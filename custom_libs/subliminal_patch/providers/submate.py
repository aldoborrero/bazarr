"""Submate provider for Bazarr - Multi-language subtitle transcription and translation.

This provider uses the Submate server to transcribe audio and translate subtitles
to ANY target language using LLM backends (Ollama, OpenAI, Claude, Gemini).

Key differences from whisperai provider:
- Supports translation to ANY language (not just English)
- Uses LLM-based translation for non-English targets
- Always sends task="transcribe" to Whisper, server handles translation
- Endpoints: /bazarr/asr, /bazarr/detect-language
"""

import functools
import logging
import time
from datetime import timedelta
from typing import TYPE_CHECKING

import ffmpeg
from babelfish.exceptions import LanguageReverseError
from pycountry import languages as py_languages
from requests import Session
from requests.exceptions import JSONDecodeError  # type: ignore[import-untyped]
from subliminal import __short_version__
from subliminal.exceptions import ConfigurationError
from subliminal.video import Episode, Movie
from subliminal_patch.providers import Provider
from subliminal_patch.subtitle import Subtitle
from subzero.language import Language

if TYPE_CHECKING:
    from subliminal.video import Video

# Whisper supported languages
# Reference: https://github.com/openai/whisper/blob/main/whisper/tokenizer.py
WHISPER_LANGUAGE_DATA = [
    ("en", "eng", "English"),
    ("zh", "zho", "Chinese"),
    ("de", "deu", "German"),
    ("es", "spa", "Spanish"),
    ("ru", "rus", "Russian"),
    ("ko", "kor", "Korean"),
    ("fr", "fra", "French"),
    ("ja", "jpn", "Japanese"),
    ("pt", "por", "Portuguese"),
    ("tr", "tur", "Turkish"),
    ("pl", "pol", "Polish"),
    ("ca", "cat", "Catalan"),
    ("nl", "nld", "Dutch"),
    ("ar", "ara", "Arabic"),
    ("sv", "swe", "Swedish"),
    ("it", "ita", "Italian"),
    ("id", "ind", "Indonesian"),
    ("hi", "hin", "Hindi"),
    ("fi", "fin", "Finnish"),
    ("vi", "vie", "Vietnamese"),
    ("he", "heb", "Hebrew"),
    ("uk", "ukr", "Ukrainian"),
    ("el", "ell", "Greek"),
    ("ms", "msa", "Malay"),
    ("cs", "ces", "Czech"),
    ("ro", "ron", "Romanian"),
    ("da", "dan", "Danish"),
    ("hu", "hun", "Hungarian"),
    ("ta", "tam", "Tamil"),
    ("no", "nor", "Norwegian"),
    ("th", "tha", "Thai"),
    ("ur", "urd", "Urdu"),
    ("hr", "hrv", "Croatian"),
    ("bg", "bul", "Bulgarian"),
    ("lt", "lit", "Lithuanian"),
    ("la", "lat", "Latin"),
    ("mi", "mri", "Maori"),
    ("ml", "mal", "Malayalam"),
    ("cy", "cym", "Welsh"),
    ("sk", "slk", "Slovak"),
    ("te", "tel", "Telugu"),
    ("fa", "fas", "Persian"),
    ("lv", "lav", "Latvian"),
    ("bn", "ben", "Bengali"),
    ("sr", "srp", "Serbian"),
    ("az", "aze", "Azerbaijani"),
    ("sl", "slv", "Slovenian"),
    ("kn", "kan", "Kannada"),
    ("et", "est", "Estonian"),
    ("mk", "mkd", "Macedonian"),
    ("br", "bre", "Breton"),
    ("eu", "eus", "Basque"),
    ("is", "isl", "Icelandic"),
    ("hy", "hye", "Armenian"),
    ("ne", "nep", "Nepali"),
    ("mn", "mon", "Mongolian"),
    ("bs", "bos", "Bosnian"),
    ("kk", "kaz", "Kazakh"),
    ("sq", "sqi", "Albanian"),
    ("sw", "swa", "Swahili"),
    ("gl", "glg", "Galician"),
    ("mr", "mar", "Marathi"),
    ("pa", "pan", "Punjabi"),
    ("si", "sin", "Sinhala"),
    ("km", "khm", "Khmer"),
    ("sn", "sna", "Shona"),
    ("yo", "yor", "Yoruba"),
    ("so", "som", "Somali"),
    ("af", "afr", "Afrikaans"),
    ("oc", "oci", "Occitan"),
    ("ka", "kat", "Georgian"),
    ("be", "bel", "Belarusian"),
    ("tg", "tgk", "Tajik"),
    ("sd", "snd", "Sindhi"),
    ("gu", "guj", "Gujarati"),
    ("am", "amh", "Amharic"),
    ("yi", "yid", "Yiddish"),
    ("lo", "lao", "Lao"),
    ("uz", "uzb", "Uzbek"),
    ("fo", "fao", "Faroese"),
    ("ht", "hat", "Haitian Creole"),
    ("ps", "pus", "Pashto"),
    ("tk", "tuk", "Turkmen"),
    ("nn", "nno", "Nynorsk"),
    ("mt", "mlt", "Maltese"),
    ("sa", "san", "Sanskrit"),
    ("lb", "ltz", "Luxembourgish"),
    ("my", "mya", "Myanmar"),
    ("bo", "bod", "Tibetan"),
    ("tl", "tgl", "Tagalog"),
    ("mg", "mlg", "Malagasy"),
    ("as", "asm", "Assamese"),
    ("tt", "tat", "Tatar"),
    ("haw", "haw", "Hawaiian"),
    ("ln", "lin", "Lingala"),
    ("ha", "hau", "Hausa"),
    ("ba", "bak", "Bashkir"),
    ("jw", "jav", "Javanese"),
    ("su", "sun", "Sundanese"),
    # Mapped languages (not directly supported by Whisper)
    ("gsw", "gsw", "Swiss German"),
]

# Language mapping for unsupported variants
LANGUAGE_MAPPING = {
    "gsw": "deu",  # Swiss German -> German
    "und": "eng",  # Undefined -> English (fallback)
}

# Ambiguous language codes that require detection
AMBIGUOUS_LANGUAGE_CODES = [
    "alg",  # Algonquian languages (family)
    "art",  # Artificial languages
    "ath",  # Athapascan languages (family)
    "aus",  # Australian languages (family)
    "mis",  # Miscellaneous languages
    "mul",  # Multiple languages
    "sgn",  # Sign languages
    "und",  # Undetermined
    "zxx",  # No linguistic content
]

logger = logging.getLogger(__name__)


def set_log_level(level: str = "INFO") -> None:
    """Set the logger level."""
    level = level.upper()
    logger.setLevel(getattr(logging, level))


class LanguageManager:
    """Manager for Whisper language data lookups."""

    def __init__(self, language_data: list[tuple[str, str, str]]) -> None:
        """Initialize with language data as list of tuples (alpha2, alpha3, name)."""
        self.language_data = language_data
        self._build_indices()

    def _build_indices(self) -> None:
        """Build lookup dictionaries for quick access."""
        self.by_alpha2: dict[str, tuple[str, str, str]] = {item[0]: item for item in self.language_data}
        self.by_alpha3: dict[str, tuple[str, str, str]] = {item[1]: item for item in self.language_data}
        self.by_name: dict[str, tuple[str, str, str]] = {item[2].lower(): item for item in self.language_data}

    def get_by_alpha2(self, code: str) -> tuple[str, str, str] | None:
        """Get language tuple by alpha2 code."""
        return self.by_alpha2.get(code)

    def get_by_alpha3(self, code: str) -> tuple[str, str, str] | None:
        """Get language tuple by alpha3 code."""
        return self.by_alpha3.get(code)

    def alpha2_to_alpha3(self, code: str) -> str | None:
        """Convert alpha2 to alpha3."""
        lang_tuple = self.get_by_alpha2(code)
        return lang_tuple[1] if lang_tuple else None

    def alpha3_to_alpha2(self, code: str | None) -> str | None:
        """Convert alpha3 to alpha2."""
        if code is None:
            return None
        lang_tuple = self.get_by_alpha3(code)
        return lang_tuple[0] if lang_tuple else None

    def get_name(self, code: str | None, code_type: str = "alpha3") -> str | None:
        """Get language name from code."""
        if code is None:
            return None
        if code_type == "alpha2":
            lang_tuple = self.get_by_alpha2(code)
        else:
            lang_tuple = self.get_by_alpha3(code)
        return lang_tuple[2] if lang_tuple else None


class WhisperLanguageManager(LanguageManager):
    """Language manager with Whisper-specific functionality."""

    def _get_language(self, code: str, name: str) -> Language | None:
        """Get Language object from code and name."""
        if code == "und":
            logger.warning("Undefined language code detected")
            return None
        try:
            return Language.fromalpha2(code)
        except LanguageReverseError:
            try:
                return Language.fromname(name)
            except LanguageReverseError:
                logger.error(f"Could not convert language: {code} ({name})")
                return None

    def get_all_language_objects(self) -> set[Language]:
        """Return set of all Language objects supported by Whisper."""
        languages = set()
        for item in self.language_data:
            lang = self._get_language(item[0], item[2])
            if lang:
                languages.add(lang)
        return languages

    def get_iso_639_2_code(self, iso639_3_code: str) -> str:
        """Get ISO 639-2 code for ffmpeg compatibility.

        ffmpeg uses older ISO 639-2 codes (e.g., 'ger' instead of 'deu').
        """
        language = py_languages.get(alpha_3=iso639_3_code)
        if language and hasattr(language, "bibliographic"):
            iso639_2_code: str = language.bibliographic
            if iso639_2_code != iso639_3_code:
                logger.debug(f"ffmpeg using language code '{iso639_2_code}' (instead of '{iso639_3_code}')")
            return iso639_2_code
        return iso639_3_code


# Global language manager instance
wlm = WhisperLanguageManager(WHISPER_LANGUAGE_DATA)


class SubmateSubtitle(Subtitle):
    """Submate subtitle representation."""

    provider_name = "submate"
    hash_verifiable = False

    def __init__(self, language: Language, video: "Video") -> None:
        super().__init__(language)
        self.video = video
        self.task: str | None = None
        self.audio_language: str | None = None
        self.force_audio_stream: str | None = None
        self.target_language: str | None = None

    @property
    def id(self) -> str:
        """Construct unique subtitle ID."""
        return f"{self.video.original_name}_{self.task}_{self.audio_language}_{self.language}"

    def get_matches(self, video: "Video") -> set[str]:
        """Get matching attributes for subtitle scoring."""
        if isinstance(video, Episode):
            self.matches.update(["series", "season", "episode"])
        elif isinstance(video, Movie):
            self.matches.update(["title"])
        return self.matches  # type: ignore[no-any-return]


class SubmateProvider(Provider):
    """Submate provider for multi-language transcription and translation.

    Unlike the whisperai provider which only supports translation to English,
    this provider supports translation to ANY language using LLM backends.

    The translation is handled by the Submate server, which:
    1. Transcribes audio using Whisper (in source language)
    2. Translates subtitles using LLM (to any target language)
    """

    provider_name = "submate"
    languages = wlm.get_all_language_objects()
    video_types = (Episode, Movie)

    def __init__(
        self,
        endpoint: str | None = None,
        response: int | None = None,
        timeout: int | None = None,
        ffmpeg_path: str | None = None,
        pass_video_name: bool | None = None,
        loglevel: str | None = None,
    ) -> None:
        """Initialize the Submate provider.

        Args:
            endpoint: Submate server URL (e.g., http://localhost:9000)
            response: Connection/response timeout in seconds
            timeout: Transcription/translation timeout in seconds
            ffmpeg_path: Path to ffmpeg binary
            pass_video_name: Whether to pass video filename to server
            loglevel: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        set_log_level(loglevel or "INFO")

        if not endpoint:
            raise ConfigurationError("Submate endpoint must be provided")
        if not response:
            raise ConfigurationError("Response timeout must be provided")
        if not timeout:
            raise ConfigurationError("Transcription timeout must be provided")
        if not ffmpeg_path:
            raise ConfigurationError("ffmpeg path must be provided")
        if pass_video_name is None:
            raise ConfigurationError("pass_video_name option must be provided")

        self.endpoint = endpoint.rstrip("/")
        self.response = int(response)
        self.timeout = int(timeout)
        self.ffmpeg_path = ffmpeg_path
        self.pass_video_name = pass_video_name
        self.session: Session | None = None

    def initialize(self) -> None:
        """Initialize the HTTP session."""
        self.session = Session()
        self.session.headers["User-Agent"] = f"Subliminal/{__short_version__}"

    def terminate(self) -> None:
        """Close the HTTP session."""
        if self.session:
            self.session.close()

    @functools.lru_cache(2)  # noqa: B019
    def encode_audio_stream(
        self,
        path: str,
        ffmpeg_path: str,
        audio_stream_language: str | None = None,
    ) -> bytes | None:
        """Encode audio stream to WAV format using ffmpeg.

        Args:
            path: Path to video file
            ffmpeg_path: Path to ffmpeg binary
            audio_stream_language: Optional language code for audio stream selection

        Returns:
            WAV audio bytes or None if encoding failed
        """
        logger.debug("Encoding audio stream to WAV with ffmpeg")

        try:
            inp = ffmpeg.input(path, threads=0)

            if audio_stream_language:
                # Use ISO 639-2 code for ffmpeg compatibility
                audio_stream_language = wlm.get_iso_639_2_code(audio_stream_language)
                logger.debug(f"Using '{audio_stream_language}' audio stream for {path}")
                lang_map = f"0:a:m:language:{audio_stream_language}"
                out = inp.output(
                    "-",
                    format="s16le",
                    acodec="pcm_s16le",
                    ac=1,
                    ar=16000,
                    af="aresample=async=1",
                    map=lang_map,
                )
            else:
                out = inp.output(
                    "-",
                    format="s16le",
                    acodec="pcm_s16le",
                    ac=1,
                    ar=16000,
                    af="aresample=async=1",
                )

            start_time = time.time()
            audio_data, _ = out.run(
                cmd=[ffmpeg_path, "-nostdin"],
                capture_stdout=True,
                capture_stderr=True,
            )
            elapsed_time = time.time() - start_time
            logger.debug(f"Encoded audio stream in {elapsed_time:.2f}s for '{path}'")

        except ffmpeg.Error as e:
            logger.warning(f"ffmpeg failed to load audio: {e.stderr.decode()}")
            return None

        logger.debug(f"Audio stream length: {len(audio_data):,} bytes")
        return bytes(audio_data)

    @functools.lru_cache(2048)  # noqa: B019
    def detect_language(self, path: str) -> Language | None:
        """Detect audio language using Submate server.

        Args:
            path: Path to video file

        Returns:
            Detected Language or None if detection failed
        """
        audio_data = self.encode_audio_stream(path, self.ffmpeg_path)
        if audio_data is None:
            logger.info(f"Cannot detect language for '{path}' - bad audio stream")
            return None

        try:
            video_name = path if self.pass_video_name else None
            assert self.session is not None, "Provider not initialized"
            r = self.session.post(
                f"{self.endpoint}/bazarr/detect-language",
                params={"encode": "false", "video_file": video_name},
                files={"audio_file": audio_data},
                timeout=(self.response, self.timeout),
            )
            results = r.json()
        except JSONDecodeError:
            logger.error("Invalid JSON response in language detection")
            return None

        if not results.get("language_code"):
            logger.info("Submate returned empty language code")
            return None

        if results["language_code"] == "und":
            logger.info("Submate detected undefined language")
            return None

        logger.debug(f"Detection results: {results}")
        return wlm._get_language(results["language_code"], results["detected_language"])

    def _determine_audio_language(
        self,
        video: "Video",
        subtitle: SubmateSubtitle,
    ) -> bool:
        """Determine audio language for transcription.

        Args:
            video: Video object
            subtitle: Subtitle to configure

        Returns:
            True if audio language was determined, False on error
        """
        if not video.audio_languages:
            # No audio language tags, run detection
            logger.debug("No audio language tags, running detection")
            detected_lang = self.detect_language(video.original_path)
            if not detected_lang:
                subtitle.task = "error"
                subtitle.release_info = "Language detection failed"
                return False

            detected_alpha3 = detected_lang.alpha3
            if detected_alpha3 in LANGUAGE_MAPPING:
                detected_alpha3 = LANGUAGE_MAPPING[detected_alpha3]
                logger.debug(f"Mapped detected language to {detected_alpha3}")

            subtitle.audio_language = detected_alpha3
            return True

        # Process audio languages with mapping
        processed_languages: dict[str, str] = {}
        for lang in video.audio_languages:
            mapped_lang = LANGUAGE_MAPPING.get(lang, lang)
            if mapped_lang != lang:
                logger.debug(f"Mapping audio language: {lang} -> {mapped_lang}")
            processed_languages[lang] = mapped_lang

        # Find matching language
        for original_lang, processed_lang in processed_languages.items():
            if subtitle.language.alpha3 == processed_lang:
                subtitle.audio_language = processed_lang
                if len(video.audio_languages) > 1:
                    subtitle.force_audio_stream = original_lang
                return True

        # No match, use first available language
        if processed_languages:
            first_lang = list(processed_languages.values())[0]
            subtitle.audio_language = first_lang

        if not subtitle.audio_language:
            subtitle.task = "error"
            subtitle.release_info = "No valid audio language"
            return False

        # Check for ambiguous language codes
        original_ambiguous = any(lang in AMBIGUOUS_LANGUAGE_CODES for lang in video.audio_languages)

        if original_ambiguous:
            logger.debug("Ambiguous language codes detected, forcing detection")
            detected_lang = self.detect_language(video.original_path)
            if not detected_lang:
                subtitle.task = "error"
                subtitle.release_info = "Bad/missing audio track"
                return False

            detected_alpha3 = detected_lang.alpha3
            if detected_alpha3 in LANGUAGE_MAPPING:
                detected_alpha3 = LANGUAGE_MAPPING[detected_alpha3]

            subtitle.audio_language = detected_alpha3

        return True

    def query(self, language: Language, video: "Video") -> SubmateSubtitle | None:
        """Query for subtitle availability.

        This is the main method called by Bazarr to check if subtitles
        can be generated for a video in the requested language.

        KEY DIFFERENCE FROM WHISPERAI:
        - whisperai rejects non-English translation targets
        - Submate accepts ANY target language (LLM handles translation)

        Args:
            language: Requested subtitle language
            video: Video object

        Returns:
            SubmateSubtitle or None if not possible
        """
        import os

        logger.debug(
            f"Query: language={language.alpha3} "
            f"({wlm.get_name(language.alpha3)}) "
            f"file='{os.path.basename(video.original_path)}'"
        )

        if language not in self.languages:
            logger.debug(f"Language {language.alpha3} not supported by Whisper")
            return None

        sub = SubmateSubtitle(language, video)
        sub.task = "transcribe"
        sub.target_language = language.alpha3

        # Determine audio language
        if not self._determine_audio_language(video, sub):
            return sub if sub.task == "error" else None

        # Determine if translation is needed
        # NOTE: Unlike whisperai, we NEVER reject based on target language
        # Submate server handles translation to ANY language via LLM
        if sub.audio_language != language.alpha3:
            sub.task = "translate"
            audio_name = wlm.get_name(sub.audio_language) or sub.audio_language
            target_name = wlm.get_name(language.alpha3) or language.alpha3
            sub.release_info = f"translate {audio_name} -> {target_name} (LLM)"
        else:
            audio_name = wlm.get_name(sub.audio_language) or sub.audio_language
            sub.release_info = f"transcribe {audio_name} audio -> SRT"

        logger.debug(f"Query result: task={sub.task} audio={sub.audio_language} -> target={language.alpha3}")
        return sub

    def list_subtitles(self, video: "Video", languages: set[Language]) -> list[SubmateSubtitle]:
        """List available subtitles for video.

        Args:
            video: Video object
            languages: Set of requested languages

        Returns:
            List of available SubmateSubtitle objects
        """
        import os

        lang_list = ", ".join(f"{lang.alpha3} ({wlm.get_name(lang.alpha3)})" for lang in languages)
        logger.debug(f"Languages requested: {lang_list} file='{os.path.basename(video.original_path)}'")
        subtitles = [self.query(lang, video) for lang in languages]
        return [s for s in subtitles if s is not None]

    def download_subtitle(self, subtitle: SubmateSubtitle) -> None:
        """Download/generate subtitle.

        This calls the Submate server to:
        1. Transcribe audio using Whisper
        2. Translate subtitles using LLM (if target != source language)

        Args:
            subtitle: Subtitle to generate
        """
        if subtitle.task == "error":
            return

        audio_data = self.encode_audio_stream(
            subtitle.video.original_path,
            self.ffmpeg_path,
            subtitle.force_audio_stream,
        )
        if not audio_data:
            logger.info(f"Cannot process {subtitle.video.original_path} - missing/bad audio track")
            subtitle.content = None
            return

        # Get target language code for subtitles
        target_alpha2 = wlm.alpha3_to_alpha2(subtitle.target_language)
        if target_alpha2 is None:
            # Fallback to alpha3 if alpha2 not found
            target_alpha2 = subtitle.target_language

        audio_name = wlm.get_name(subtitle.audio_language) or subtitle.audio_language
        target_name = wlm.get_name(subtitle.target_language) or subtitle.target_language

        logger.info(
            f"Starting transcription: {audio_name} audio -> {target_name} subtitles for {subtitle.video.original_path}"
        )

        start_time = time.time()
        video_name = subtitle.video.original_path if self.pass_video_name else None

        # Call Submate server
        # - task="transcribe": Always transcribe (Whisper's translate only does English)
        # - language: Target language for subtitles (server handles LLM translation)
        assert self.session is not None, "Provider not initialized"
        r = self.session.post(
            f"{self.endpoint}/bazarr/asr",
            params={
                "task": "transcribe",  # Always transcribe, server handles translation
                "language": target_alpha2,  # Target language for output
                "output": "srt",
                "encode": "false",  # Audio already encoded by ffmpeg
                "video_file": video_name,
            },
            files={"audio_file": audio_data},
            timeout=(self.response, self.timeout),
        )

        elapsed = timedelta(seconds=round(time.time() - start_time))

        # Log response info
        subtitle_length = len(r.content)
        logger.debug(f"Returned subtitle length: {subtitle_length:,} bytes")
        if subtitle_length > 0:
            preview = r.content[: min(subtitle_length, 500)]
            logger.debug(f"Subtitle preview: {preview}")

        logger.info(f"Completed in {elapsed}: {audio_name} -> {target_name} for {subtitle.video.original_path}")

        subtitle.content = r.content
