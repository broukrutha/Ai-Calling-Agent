"""
Make model output fit to be spoken.

Language models write for screens: Markdown emphasis, bullet lists, headings,
links, emoji and table pipes. A text-to-speech engine reads all of that
literally ("asterisk asterisk note asterisk asterisk"). These functions turn
it into plain sentences before synthesis, and clean the text shown in the chat.
"""

from __future__ import annotations

import re

_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`]*)`")
_MD_LINK = re.compile(r"\[([^\]]+)\]\((?:[^)]+)\)")
_URL = re.compile(r"https?://([^/\s]+)[^\s]*")
_EMPHASIS = re.compile(r"(\*\*|__|\*|_)(?=\S)(.+?)(?<=\S)\1")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.MULTILINE)
_TABLE_RULE = re.compile(r"^\s*\|?[\s:|-]{3,}\|?\s*$", re.MULTILINE)
_CITATION = re.compile(r"\s*\[(?:\d+(?:,\s*\d+)*)\]")
_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF\U0000FE0F\U0000200D]"
)
_RUPEE_AMOUNT = re.compile(r"(?:₹|\bRs\.?|\bINR)\s?(\d[\d,]*(?:\.\d+)?)")
_SPACES = re.compile(r"[ \t]{2,}")


def clean_markdown(text: str) -> str:
    """Remove Markdown and citation markers, keeping the words. For display."""
    text = _CODE_FENCE.sub(" ", text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _EMPHASIS.sub(r"\2", text)
    text = _HEADING.sub("", text)
    text = _TABLE_RULE.sub("", text)
    text = text.replace("|", " ")
    text = _CITATION.sub("", text)
    return _SPACES.sub(" ", text).strip()


def to_speech(text: str) -> str:
    """Plain sentences for text-to-speech."""
    text = clean_markdown(text)
    text = _URL.sub(lambda match: match.group(1).removeprefix("www."), text)
    text = _RUPEE_AMOUNT.sub(r"\1 rupees", text)
    text = _EMOJI.sub("", text)
    # Each bullet becomes its own sentence so the voice pauses between items.
    lines = []
    for line in text.splitlines():
        line = _BULLET.sub("", line).strip()
        if line:
            lines.append(line if line.endswith((".", "!", "?", "।", ":", ",")) else line + ".")
    return _SPACES.sub(" ", " ".join(lines)).strip()
