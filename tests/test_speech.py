import pytest

from voice_agent.speech import clean_markdown, to_speech


@pytest.mark.parametrize("raw,spoken", [
    ("**Cleaning** costs ₹1,200.", "Cleaning costs 1,200 rupees."),
    ("## Hours\n- Monday to Friday: 9 to 7\n- Sunday: closed", "Hours. Monday to Friday: 9 to 7. Sunday: closed."),
    ("See [our site](https://example.com/pricing) for details [1].", "See our site for details."),
    ("Visit https://www.example.com/a/b today", "Visit example.com today."),
    ("Great news 🎉 we are open", "Great news we are open."),
    ("Rs. 30 per hour", "30 rupees per hour."),
    ("`code` and __bold__", "code and bold."),
    ("", ""),
])
def test_to_speech(raw, spoken):
    assert to_speech(raw) == spoken


def test_clean_markdown_keeps_line_breaks_and_numbers():
    assert clean_markdown("**1,200** rupees [2]") == "1,200 rupees"
    assert clean_markdown("| a | b |\n|---|---|\n| 1 | 2 |") == "a b \n\n 1 2"
