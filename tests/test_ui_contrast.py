"""The console's colour tokens keep text readable (WCAG 2.x AA) in both themes."""

import re
from pathlib import Path

import pytest

CSS = (Path(__file__).resolve().parent.parent / "static" / "style.css").read_text(encoding="utf-8")


def _tokens(block: str) -> dict[str, str]:
    return dict(re.findall(r"--([\w-]+):\s*(#[0-9a-fA-F]{6})", block))


def _block(selector_pattern: str) -> dict[str, str]:
    match = re.search(selector_pattern + r"\s*\{(.*?)\n\}", CSS, re.S)
    assert match, selector_pattern
    return _tokens(match.group(1))


LIGHT = _block(r"\n:root")
DARK_OVERRIDES = _block(r"\n:root\[data-theme='dark'\]")
DARK = {**LIGHT, **DARK_OVERRIDES}          # what the cascade resolves in dark mode


def _luminance(hex_colour: str) -> float:
    channels = [int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast(a: str, b: str) -> float:
    high, low = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


TEXT_PAIRS = [
    (fg, bg)
    for fg in ("text", "text-2", "text-3")
    for bg in ("canvas", "bg", "surface", "surface-2", "surface-3")
] + [
    ("accent-text", "surface"), ("accent-text", "accent-soft"),
    ("good", "good-soft"), ("good", "surface"), ("warn", "warn-soft"), ("warn", "surface"),
    ("bad", "bad-soft"), ("bad", "surface"),
]


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_text_meets_aa_on_every_surface(theme):
    tokens = LIGHT if theme == "light" else DARK
    failures = [
        f"{fg} on {bg}: {contrast(tokens[fg], tokens[bg]):.2f}"
        for fg, bg in TEXT_PAIRS
        if contrast(tokens[fg], tokens[bg]) < 4.5
    ]
    assert not failures, failures


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_white_labels_on_solid_buttons(theme):
    tokens = LIGHT if theme == "light" else DARK
    for solid in ("accent", "accent-hover", "bad-solid"):
        assert contrast("#ffffff", tokens[solid]) >= 4.5, solid


def test_dark_theme_is_declared_for_the_system_setting_too():
    system = re.search(r"prefers-color-scheme: dark\)\s*\{\s*:root:not\(\[data-theme='light'\]\)\s*\{(.*?)\n  \}", CSS, re.S)
    assert system and _tokens(system.group(1)) == DARK_OVERRIDES


def test_stylesheet_matches_the_palette_generator():
    import importlib.util

    path = Path(__file__).resolve().parent.parent / "scripts" / "generate_palette.py"
    spec = importlib.util.spec_from_file_location("generate_palette", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for theme, tokens in (("light", LIGHT), ("dark", DARK)):
        generated = module.semantic(theme)
        drift = {name: (tokens.get(name), value) for name, value in generated.items() if tokens.get(name) != value}
        assert not drift, f"{theme}: style.css differs from scripts/generate_palette.py: {drift}"
