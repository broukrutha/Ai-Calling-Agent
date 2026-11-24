"""
Generate the web console's colour tokens and check their contrast.

    python scripts/generate_palette.py

The palette is two 12-step scales, a cool neutral and an indigo accent, built
in OKLCH. Equal lightness steps look equal there, which HSL does not give.
Step roles follow Radix Colors (1 app background … 12 high-contrast text).
Values are converted to sRGB hex, with out-of-gamut colours brought in by
reducing chroma.

The script prints the CSS custom properties for both themes and a WCAG
contrast report, and exits with an error if any text pair is below AA.
``static/style.css`` holds this output; tests/test_ui_contrast.py checks that
they match.
"""

from __future__ import annotations

import math
import sys


def oklch_to_srgb(L, C, h):
    a, b = C * math.cos(math.radians(h)), C * math.sin(math.radians(h))
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    long_, medium, short = l_ ** 3, m_ ** 3, s_ ** 3      # LMS cone responses
    r = 4.0767416621 * long_ - 3.3077115913 * medium + 0.2309699292 * short
    g = -1.2684380046 * long_ + 2.6097574011 * medium - 0.3413193965 * short
    bl = -0.0041960863 * long_ - 0.7034186147 * medium + 1.7076147010 * short
    return r, g, bl


def in_gamut(rgb):
    return all(-1e-4 <= c <= 1 + 1e-4 for c in rgb)


def to_hex(L, C, h):
    c = C
    rgb = oklch_to_srgb(L, c, h)
    while not in_gamut(rgb) and c > 0:
        c -= 0.002
        rgb = oklch_to_srgb(L, max(c, 0), h)
    out = []
    for v in rgb:
        v = min(max(v, 0.0), 1.0)
        v = 12.92 * v if v <= 0.0031308 else 1.055 * v ** (1 / 2.4) - 0.055
        out.append(round(v * 255))
    return "#{:02x}{:02x}{:02x}".format(*out)


def luminance(hex_):
    rgb = [int(hex_[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a, b):
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


# ── Scales: (L, C) per step, hue per scale ────────────────────────────────────
NEUTRAL_H = 264
ACCENT_H = 277      # indigo, between Linear's indigo and Radix iris

LIGHT = {
    "gray": [(.995, .0015), (.978, .003), (.962, .004), (.947, .005), (.932, .006), (.912, .007),
             (.875, .009), (.79, .012), (.64, .015), (.6, .015), (.52, .017), (.225, .02)],
    "accent": [(.99, .008), (.972, .018), (.95, .035), (.925, .055), (.895, .075), (.855, .095),
               (.8, .12), (.72, .155), (.54, .205), (.49, .2), (.48, .18), (.3, .11)],
}
DARK = {
    "gray": [(.165, .004), (.185, .005), (.215, .006), (.24, .007), (.262, .008), (.29, .009),
             (.335, .011), (.42, .013), (.52, .015), (.57, .015), (.765, .012), (.955, .004)],
    "accent": [(.19, .03), (.215, .045), (.27, .085), (.31, .11), (.35, .13), (.4, .14),
               (.46, .155), (.55, .175), (.565, .2), (.515, .19), (.81, .11), (.94, .04)],
}
STATUS = {           # hue, light (soft L,C; text L,C; solid L,C), dark (…)
    "good": (155, [(.965, .035), (.47, .12), (.6, .15)], [(.25, .05), (.79, .15), (.66, .16)]),
    "warn": (65, [(.97, .045), (.5, .12), (.75, .16)], [(.26, .05), (.82, .13), (.74, .15)]),
    "bad": (25, [(.965, .03), (.52, .19), (.565, .2)], [(.25, .06), (.75, .15), (.575, .2)]),
}


def build(scales):
    return {name: [to_hex(L, C, ACCENT_H if name == "accent" else NEUTRAL_H) for L, C in steps]
            for name, steps in scales.items()}


def status(mode):
    out = {}
    for name, (h, light, dark) in STATUS.items():
        steps = light if mode == "light" else dark
        out[name] = [to_hex(L, C, h) for L, C in steps]
    return out


def semantic(mode: str) -> dict[str, str]:
    """Semantic tokens (CSS custom-property name → hex) for one theme."""
    s = build(LIGHT if mode == "light" else DARK)
    g, a = s["gray"], s["accent"]
    st = status(mode)
    if mode == "light":
        tokens = {
            "canvas": g[1], "bg": g[0], "surface": "#ffffff", "surface-2": g[2], "surface-3": g[3], "surface-4": g[4],
            "border": to_hex(.92, .007, NEUTRAL_H), "border-strong": g[6], "border-heavy": g[7],
            "text": g[11], "text-2": to_hex(.47, .018, NEUTRAL_H), "text-3": to_hex(.515, .016, NEUTRAL_H),
            "accent": a[8], "accent-hover": a[9], "accent-text": a[10], "accent-soft": to_hex(.96, .03, ACCENT_H), "accent-soft-2": a[3],
            "accent-border": a[6], "on-accent": "#ffffff",
        }
    else:
        tokens = {
            "canvas": g[0], "bg": g[1], "surface": g[2], "surface-2": g[3], "surface-3": g[4], "surface-4": g[5],
            "border": to_hex(.272, .008, NEUTRAL_H), "border-strong": to_hex(.315, .01, NEUTRAL_H), "border-heavy": g[7],
            "text": g[11], "text-2": g[10], "text-3": to_hex(.68, .012, NEUTRAL_H),
            "accent": a[8], "accent-hover": a[9], "accent-text": a[10], "accent-soft": a[2], "accent-soft-2": a[3],
            "accent-border": a[6], "on-accent": "#ffffff",
        }
    for name, (soft, text, solid) in st.items():
        tokens[f"{name}-soft"], tokens[name], tokens[f"{name}-solid"] = soft, text, solid
    return tokens



def report(mode: str, t: dict[str, str]) -> list[str]:
    failures: list[str] = []

    def check(label: str, fg: str, bg: str, minimum: float) -> None:
        ratio = contrast(fg, bg)
        if ratio < minimum:
            failures.append(f"{mode}: {label}")
        print(f"  {'ok ' if ratio >= minimum else 'LOW'} {ratio:5.2f}  (>={minimum})  {label}  {fg} on {bg}")

    for surface in ("canvas", "bg", "surface", "surface-2", "surface-3"):
        check(f"text on {surface}", t["text"], t[surface], 7)
        check(f"text-2 on {surface}", t["text-2"], t[surface], 4.5)
        check(f"text-3 on {surface}", t["text-3"], t[surface], 4.5)
    check("accent-text on surface", t["accent-text"], t["surface"], 4.5)
    check("accent-text on accent-soft", t["accent-text"], t["accent-soft"], 4.5)
    for solid in ("accent", "accent-hover", "bad-solid"):
        check(f"white on {solid}", "#ffffff", t[solid], 4.5)
    for name in ("good", "warn", "bad"):
        check(f"{name} on {name}-soft", t[name], t[f"{name}-soft"], 4.5)
        check(f"{name} on surface", t[name], t["surface"], 4.5)
    check("accent focus ring vs surface", t["accent"], t["surface"], 3)
    return failures


def main() -> int:
    failures: list[str] = []
    for mode in ("light", "dark"):
        tokens = semantic(mode)
        print(f"\n/* {mode} */")
        for name, value in tokens.items():
            print(f"  --{name}: {value};")
        print(f"\n{mode} contrast (WCAG 2.x)")
        failures += report(mode, tokens)
    print("\nBelow AA:", ", ".join(failures) if failures else "none")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
