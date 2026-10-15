"""Vietnamese Post-Processor: Punctuation, Capitalization & Inverse Text Normalization (ITN).

QUAN-Transcribe's raw CTC output is lowercase Vietnamese without punctuation, with
numbers spelled out as words (e.g. "xin chào tôi tên là nam", "năm mươi phần trăm").

This module provides high-speed, rule-based and vocabulary-driven post-processing:
1. Capitalization: Capitalizes the start of sentences, proper names, and common acronyms.
2. Punctuation: Detects questions (ai, gì, ở đâu, phải không...), exclamations, pauses.
3. Inverse Text Normalization (ITN): Converts spoken Vietnamese numbers into digits,
   percentages, currencies, and dates.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

try:
    from sea_g2p import G2P, Normalizer
    _SEA_G2P_AVAILABLE = True
except ImportError:
    _SEA_G2P_AVAILABLE = False
    G2P = None  # type: ignore[assignment,misc]
    Normalizer = None  # type: ignore[assignment,misc]

if TYPE_CHECKING:
    from .audio import Segment

#: Common Vietnamese proper nouns & geographical locations to capitalize
_PROPER_NOUNS: dict[str, str] = {
    "việt nam": "Việt Nam",
    "hà nội": "Hà Nội",
    "hồ chí minh": "Hồ Chí Minh",
    "sài gòn": "Sài Gòn",
    "đà nẵng": "Đà Nẵng",
    "hải phòng": "Hải Phòng",
    "cần thơ": "Cần Thơ",
    "huế": "Huế",
    "nha trang": "Nha Trang",
    "vũng tàu": "Vũng Tàu",
    "quảng ninh": "Quảng Ninh",
    "bắc ninh": "Bắc Ninh",
    "đồng nai": "Đồng Nai",
    "bình dương": "Bình Dương",
    "mỹ": "Mỹ",
    "hoa kỳ": "Hoa Kỳ",
    "anh": "Anh",
    "pháp": "Pháp",
    "đức": "Đức",
    "nhật bản": "Nhật Bản",
    "hàn quốc": "Hàn Quốc",
    "trung quốc": "Trung Quốc",
    "thái lan": "Thái Lan",
    "singapore": "Singapore",
    "đông nam á": "Đông Nam Á",
    "châu á": "Châu Á",
    "châu âu": "Châu Âu",
}

#: Common Vietnamese acronyms and tech / company brands
_ACRONYMS: dict[str, str] = {
    "ai": "AI",
    "asr": "ASR",
    "ctc": "CTC",
    "gdp": "GDP",
    "cpu": "CPU",
    "gpu": "GPU",
    "ram": "RAM",
    "ssd": "SSD",
    "api": "API",
    "usd": "USD",
    "vnd": "VNĐ",
    "vtv": "VTV",
    "vov": "VOV",
    "fpt": "FPT",
    "vnpt": "VNPT",
    "viettel": "Viettel",
    "vingroup": "Vingroup",
    "google": "Google",
    "openai": "OpenAI",
    "microsoft": "Microsoft",
    "facebook": "Facebook",
    "youtube": "YouTube",
    "zalo": "Zalo",
}

#: Vietnamese digits mapping
_DIGITS: dict[str, int] = {
    "không": 0,
    "một": 1,
    "mốt": 1,
    "hai": 2,
    "ba": 3,
    "bốn": 4,
    "tư": 4,
    "năm": 5,
    "lăm": 5,
    "sáu": 6,
    "bảy": 7,
    "bẩy": 7,
    "tám": 8,
    "chín": 9,
}

_SCALES: dict[str, int] = {
    "mười": 10,
    "mươi": 10,
    "trăm": 100,
    "nghìn": 1000,
    "ngàn": 1000,
    "triệu": 1000000,
    "tỷ": 1000000000,
    "tiền": 1000000000,
}

#: Question triggers at the end of sentences
_QUESTION_ENDS = re.compile(
    r"\b(phải không|đúng không|được không|chưa|hả|nhỉ|à|sao|không|thế|sao ạ)\s*$",
    re.IGNORECASE,
)

#: Question triggers at the beginning of sentences
_QUESTION_STARTS = re.compile(
    r"^\b(ai|cái gì|gì|ở đâu|đâu|khi nào|bao giờ|tại sao|vì sao|làm sao|như thế nào|thế nào|bao nhiêu|mấy)\b",
    re.IGNORECASE,
)


def _format_int(n: int) -> str:
    """Format large integers with dots (Vietnamese standard, e.g. 1.000.000)."""
    return f"{n:,}".replace(",", ".")


def _words_to_number(words: list[str]) -> int | None:
    """Parse a sequence of Vietnamese number words into an integer."""
    if not words:
        return None
    # Special single digit check
    if len(words) == 1 and words[0] in _DIGITS:
        return _DIGITS[words[0]]

    total = 0
    current = 0
    i = 0
    matched_any = False

    while i < len(words):
        w = words[i]
        if w in _DIGITS:
            matched_any = True
            # Special check for "không trăm"
            if w == "không" and i + 1 < len(words) and words[i + 1] == "trăm":
                i += 2  # skip both 'không' and 'trăm'
                continue
            current += _DIGITS[w]
            i += 1
        elif w in ("mười", "mươi"):
            matched_any = True
            if current == 0:
                current = 10
            else:
                current *= 10
            i += 1
        elif w == "trăm":
            matched_any = True
            if current == 0:
                current = 100
            else:
                current *= 100
            i += 1
        elif w in ("nghìn", "ngàn"):
            matched_any = True
            if current == 0:
                current = 1
            total += current * 1000
            current = 0
            i += 1
        elif w == "triệu":
            matched_any = True
            if current == 0:
                current = 1
            total += current * 1000000
            current = 0
            i += 1
        elif w in ("tỷ", "tiền"):
            matched_any = True
            if current == 0:
                current = 1
            total += current * 1000000000
            current = 0
            i += 1
        elif w in ("linh", "lẻ"):
            i += 1
        else:
            return None

    if not matched_any:
        return None
    return total + current


def inverse_text_normalization(text: str) -> str:
    """Convert Vietnamese spoken numbers, percentages and currencies into digits.

    Examples:
    - "năm mươi phần trăm" -> "50%"
    - "hai mươi mốt" -> "21"
    - "năm trăm nghìn đồng" -> "500.000 VNĐ"
    - "ngày mười lăm tháng mười năm hai nghìn không trăm hai mươi sáu" -> "ngày 15/10/2026"
    """
    if not text:
        return text

    # Base list of number words
    num_word = (
        r"(?:không|một|mốt|hai|ba|bốn|tư|năm|lăm|sáu|bảy|bẩy|tám|chín|mười|mươi|"
        r"trăm|nghìn|ngàn|triệu|tỷ|linh|lẻ)"
    )
    multi_num = rf"(?:{num_word}(?:\s+{num_word})*)"
    month_word = r"(?:mười\s+(?:một|hai)|mười|một|hai|ba|bốn|tư|năm|sáu|bảy|tám|chín|\d+)"

    # 1. Full Date expression: "ngày X tháng Y năm Z" or "ngày X tháng Y"
    def _replace_full_date(m: re.Match[str]) -> str:
        d_str = m.group(1).strip()
        m_str = m.group(2).strip()
        y_str = m.group(3).strip() if m.group(3) else ""

        d_val = _words_to_number(d_str.split()) or d_str
        m_val = _words_to_number(m_str.split()) or m_str
        if y_str:
            y_val = _words_to_number(y_str.split()) or y_str
            return f"ngày {d_val}/{m_val}/{y_val}"
        return f"ngày {d_val}/{m_val}"

    date_full_pat = rf"\bngày\s+({multi_num})\s+tháng\s+({month_word})(?:\s+năm\s+({multi_num}))?\b"
    text = re.sub(date_full_pat, _replace_full_date, text, flags=re.IGNORECASE)

    # 2. Standalone calendar year: only "năm hai nghìn...", "năm một nghìn chín trăm..."
    # Disambiguation: "năm trăm", "năm mươi", "năm nghìn" is number 5, not year!
    def _replace_year(m: re.Match[str]) -> str:
        y_str = m.group(1).strip()
        first_word = y_str.split()[0]
        # If it's "năm trăm" or "năm mươi", it's the number 5, do not convert as year!
        if first_word in ("mươi", "mười", "trăm", "nghìn", "ngàn", "triệu", "tỷ"):
            return m.group(0)
        val = _words_to_number(y_str.split())
        if val is not None and 1900 <= val <= 2100:
            return f"năm {val}"
        return m.group(0)

    text = re.sub(rf"\bnăm\s+({multi_num})\b", _replace_year, text, flags=re.IGNORECASE)

    # 3. Percentages: "X phần trăm" -> "X%"
    def _replace_percent(m: re.Match[str]) -> str:
        w_str = m.group(1).strip()
        val = _words_to_number(w_str.split())
        if val is not None:
            return f"{val}%"
        return f"{w_str}%"

    text = re.sub(
        rf"\b({multi_num}|[\d\.\,]+)\s+phần\s+trăm\b",
        _replace_percent,
        text,
        flags=re.IGNORECASE,
    )

    # 4. Currency: "X đồng" -> "X VNĐ"
    def _replace_currency(m: re.Match[str]) -> str:
        w = m.group(1).strip().split()
        val = _words_to_number(w)
        if val is not None:
            return f"{_format_int(val)} VNĐ"
        return m.group(0)

    curr_pat = rf"\b({multi_num})\s+đồng\b"
    text = re.sub(curr_pat, _replace_currency, text, flags=re.IGNORECASE)

    # 5. Decimals: "sáu phẩy năm" -> "6,5"
    def _replace_decimal(m: re.Match[str]) -> str:
        w1 = m.group(1).strip().split()
        w2 = m.group(2).strip().split()
        v1 = _words_to_number(w1)
        v2 = _words_to_number(w2)
        if v1 is not None and v2 is not None:
            return f"{v1},{v2}"
        return m.group(0)

    dec_pat = rf"\b({multi_num})\s+phẩy\s+({multi_num})\b"
    text = re.sub(dec_pat, _replace_decimal, text, flags=re.IGNORECASE)

    # 6. General compound numbers (2 or more words)
    multi_num_strict = rf"\b({num_word}(?:\s+{num_word})+)\b"

    def _replace_multi_num(m: re.Match[str]) -> str:
        words = m.group(1).strip().split()
        val = _words_to_number(words)
        if val is not None and len(words) >= 2:
            return _format_int(val) if val >= 1000 else str(val)
        return m.group(0)

    text = re.sub(multi_num_strict, _replace_multi_num, text, flags=re.IGNORECASE)

    return text


def capitalize_vietnamese(text: str) -> str:
    """Capitalize sentence beginnings, proper names, and common acronyms."""
    if not text:
        return text

    # First, handle proper nouns and acronyms (case-insensitive replace)
    words = text.split()
    if not words:
        return text

    # Replace multi-word proper nouns first
    for name_lower, name_styled in sorted(
        _PROPER_NOUNS.items(), key=lambda x: len(x[0]), reverse=True
    ):
        pattern = rf"\b{re.escape(name_lower)}\b"
        text = re.sub(pattern, name_styled, text, flags=re.IGNORECASE)

    # Replace acronyms
    for acr_lower, acr_styled in _ACRONYMS.items():
        pattern = rf"\b{re.escape(acr_lower)}\b"
        text = re.sub(pattern, acr_styled, text, flags=re.IGNORECASE)

    # Capitalize the first letter of each sentence (after . ? ! or start)
    def _cap_match(m: re.Match[str]) -> str:
        prefix = m.group(1)
        char = m.group(2)
        return prefix + char.upper()

    text = re.sub(r"(^|[.?!]\s+)(\w)", _cap_match, text)

    return text


def punctuate_vietnamese(text: str, is_final: bool = True) -> str:
    """Add intelligent punctuation (., ?, !) to unpunctuated Vietnamese text."""
    trimmed = text.strip()
    if not trimmed:
        return text

    # Already has ending punctuation
    if trimmed[-1] in ".?!":
        return text

    # Check for question
    is_question = bool(
        _QUESTION_ENDS.search(trimmed) or _QUESTION_STARTS.search(trimmed)
    )

    if is_final:
        return f"{trimmed}?" if is_question else f"{trimmed}."

    return trimmed


_g2p_instance: Any = None
_normalizer_instance: Any = None


def get_g2p(lang: str = "vi") -> Any:
    global _g2p_instance
    if not _SEA_G2P_AVAILABLE:
        return None
    if _g2p_instance is None:
        try:
            _g2p_instance = G2P(lang)
        except Exception as exc:
            logger.warning("Failed to initialize sea-g2p G2P: %s", exc)
            return None
    return _g2p_instance


def get_normalizer(lang: str = "vi") -> Any:
    global _normalizer_instance
    if not _SEA_G2P_AVAILABLE:
        return None
    if _normalizer_instance is None:
        try:
            _normalizer_instance = Normalizer(lang)
        except Exception as exc:
            logger.warning("Failed to initialize sea-g2p Normalizer: %s", exc)
            return None
    return _normalizer_instance


def to_phonemes(text: str, lang: str = "vi") -> str:
    """Convert Vietnamese or multilingual text to IPA phonemes using sea-g2p."""
    if not text or not text.strip():
        return ""
    g2p = get_g2p(lang)
    if g2p is None:
        return ""
    try:
        return g2p.convert(text.strip())
    except Exception as exc:
        logger.debug("sea-g2p conversion error: %s", exc)
        return ""


def text_normalization(text: str, lang: str = "vi") -> str:
    """Run Text Normalization (TN) via sea-g2p (spoken word expansion for TTS)."""
    if not text or not text.strip():
        return ""
    norm = get_normalizer(lang)
    if norm is None:
        return text
    try:
        res = norm.normalize(text.strip())
        return str(res) if res else text
    except Exception as exc:
        logger.debug("sea-g2p normalizer error: %s", exc)
        return text


class PostProcessor:
    """Complete Vietnamese speech-to-text post-processing pipeline."""

    def __init__(
        self,
        *,
        enable_punctuation: bool = True,
        enable_itn: bool = True,
        enable_capitalization: bool = True,
        enable_g2p: bool = False,
        norm_mode: str = "itn",
    ) -> None:
        self.enable_punctuation = enable_punctuation
        self.enable_itn = enable_itn
        self.enable_capitalization = enable_capitalization
        self.enable_g2p = enable_g2p
        self.norm_mode = norm_mode

    def process(self, text: str, is_final: bool = True) -> str:
        """Run the configured post-processing steps on raw transcript text."""
        result = text.strip()
        if not result:
            return result

        if self.norm_mode == "tn":
            result = text_normalization(result)
        elif self.norm_mode == "itn" and self.enable_itn:
            result = inverse_text_normalization(result)

        if self.enable_capitalization:
            result = capitalize_vietnamese(result)

        if self.enable_punctuation:
            result = punctuate_vietnamese(result, is_final=is_final)

        return result

    def process_segments(
        self,
        segments: list[Segment],
        include_phonemes: bool = False,
    ) -> list[Segment]:
        """Apply post-processing across audio segments, optionally attaching phonemes."""
        if not segments:
            return segments

        for i, seg in enumerate(segments):
            is_last = i == len(segments) - 1
            seg.text = self.process(seg.text, is_final=is_last)
            if include_phonemes or self.enable_g2p:
                seg.phonemes = to_phonemes(seg.text)

        return segments


#: Default singleton post-processor
default_post_processor = PostProcessor()

