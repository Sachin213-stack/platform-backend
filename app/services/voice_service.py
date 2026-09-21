import re
import io
import asyncio
from typing import AsyncGenerator, Optional, Dict
import edge_tts
from app.core.logging import logger

# Voice model mapping from client IDs / preferences to Edge-TTS neural models
VOICE_MAP: Dict[str, str] = {
    # Default Flagship Voice: Modern ChatGPT-style warm companion
    "friday-core-female": "en-US-AriaNeural",
    "en-US-AriaNeural": "en-US-AriaNeural",
    
    # Classic FRIDAY AI-CTO: Crisp British technical voice
    "friday-nova-neutral": "en-GB-SoniaNeural",
    "en-GB-SoniaNeural": "en-GB-SoniaNeural",
    
    # Deep, resonant engineering lead
    "friday-echo-male": "en-US-GuyNeural",
    "en-US-GuyNeural": "en-US-GuyNeural",
    
    # Natural Indian English & Hinglish partner
    "friday-solis-female": "hi-IN-SwaraNeural",
    "hi-IN-SwaraNeural": "hi-IN-SwaraNeural",
    "en-IN-NeerjaNeural": "en-IN-NeerjaNeural",
}

DEFAULT_VOICE = "en-US-AriaNeural"


def clean_text_for_speech(text: str) -> str:
    """
    Cleans and normalizes LLM text output for natural neural speech synthesis:
    - Replaces code blocks with natural spoken phrases (no square brackets)
    - Strips markdown tables, pipes, JSON curly braces, and bullet symbols
    - Converts status indicators (checkmarks, warnings) into spoken words
    - Expands technical metric notations (ms -> milliseconds, req/s -> requests per second, GB -> gigabytes)
    - Strips emojis, non-ASCII symbols, and excess whitespace
    """
    if not text:
        return ""

    t = text
    # 1. Translate common status emojis to spoken words
    t = re.sub(r"✅", " Confirmed. ", t)
    t = re.sub(r"⚠️", " Warning: ", t)
    t = re.sub(r"❌", " Error: ", t)

    # 2. Replace code blocks with natural speech (no square brackets)
    t = re.sub(r"```[\s\S]*?```", " The technical code details are provided in your session transcript. ", t)

    # 3. Strip inline code ticks
    t = re.sub(r"`([^`]+)`", r"\1", t)

    # 4. Strip markdown table divider rows (e.g. |---|---|)
    t = re.sub(r"\|[ -:|]+\|", " ", t)
    # Strip standalone table pipes
    t = re.sub(r"\|", ", ", t)

    # 5. Remove markdown headers, bold, italics, strikethroughs, blockquotes
    t = re.sub(r"^[ \t]*[#>*\-•][ \t]+", " ", t, flags=re.MULTILINE)
    t = re.sub(r"[#*_~]", "", t)

    # 6. Remove markdown links [title](url) -> title
    t = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", t)
    # Remove raw square brackets
    t = re.sub(r"[\[\]]", "", t)
    # Remove raw URLs
    t = re.sub(r"https?://\S+", "", t)

    # 7. Strip raw JSON / curly braces
    t = re.sub(r"\{[^{}]*\}", " configuration parameters ", t)
    t = re.sub(r"[{}]", " ", t)

    # 8. Convert technical shorthand and metrics for spoken naturalness
    t = re.sub(r"\bp99\b", "P 99", t, flags=re.IGNORECASE)
    t = re.sub(r"\bp95\b", "P 95", t, flags=re.IGNORECASE)
    t = re.sub(r"\bp50\b", "P 50", t, flags=re.IGNORECASE)
    t = re.sub(r"(\d+)\s*ms\b", r"\1 milliseconds", t, flags=re.IGNORECASE)
    t = re.sub(r"\bms\b", "milliseconds", t, flags=re.IGNORECASE)
    t = re.sub(r"\breq/s(?:ec)?\b", "requests per second", t, flags=re.IGNORECASE)
    t = re.sub(r"(\d+)\s*GB\b", r"\1 gigabytes", t, flags=re.IGNORECASE)
    t = re.sub(r"(\d+)\s*MB\b", r"\1 megabytes", t, flags=re.IGNORECASE)
    t = re.sub(r"(\d+)\s*KB\b", r"\1 kilobytes", t, flags=re.IGNORECASE)
    t = re.sub(r"\bvCPUs?\b", "virtual CPUs", t, flags=re.IGNORECASE)
    t = re.sub(r"\bpct\b", "percent", t, flags=re.IGNORECASE)
    t = re.sub(r"%", " percent", t)

    # 9. Clean non-ASCII symbols and normalize whitespace
    t = re.sub(r"[^\x00-\x7F]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t



class VoiceService:
    """
    High-fidelity asynchronous Text-to-Speech service powered by Microsoft Edge Neural TTS.
    Provides streaming MP3 audio with rate, pitch, and voice profile control.
    """

    @staticmethod
    def resolve_voice(voice_id: Optional[str]) -> str:
        if not voice_id:
            return DEFAULT_VOICE
        return VOICE_MAP.get(voice_id, DEFAULT_VOICE)

    @staticmethod
    def format_rate(rate: Optional[float]) -> str:
        """Converts float rate (e.g. 1.1) to Edge-TTS string '+10%'."""
        if rate is None or rate == 1.0:
            return "+0%"
        diff = int(round((rate - 1.0) * 100))
        return f"+{diff}%" if diff >= 0 else f"{diff}%"

    @staticmethod
    def format_pitch(pitch_str: Optional[str]) -> str:
        if not pitch_str:
            return "+0Hz"
        return pitch_str

    async def stream_speech(
        self,
        text: str,
        voice_id: Optional[str] = None,
        rate: Optional[float] = 1.0,
        pitch: Optional[str] = "+0Hz",
    ) -> AsyncGenerator[bytes, None]:
        """
        Synthesizes text into an MP3 audio byte stream chunk-by-chunk.
        """
        clean_text = clean_text_for_speech(text)
        if not clean_text:
            return

        selected_voice = self.resolve_voice(voice_id)
        rate_str = self.format_rate(rate)
        pitch_str = self.format_pitch(pitch)

        logger.debug(
            "Synthesizing speech via Edge-TTS (voice=%s, rate=%s, chars=%d)",
            selected_voice,
            rate_str,
            len(clean_text),
        )

        communicate = edge_tts.Communicate(
            text=clean_text,
            voice=selected_voice,
            rate=rate_str,
            pitch=pitch_str,
        )

        try:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]
        except Exception as e:
            logger.error("Edge-TTS synthesis error: %s", e)
            raise


voice_service = VoiceService()
